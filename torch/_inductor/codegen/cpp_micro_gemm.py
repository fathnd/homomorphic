from collections import namedtuple
from typing import Dict, List, Optional, Type

import sympy

import torch

from .. import ir
from ..codecache import pick_vec_isa, VecAVX2, VecAVX512
from ..utils import IndentedBuffer, parallel_num_threads
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, value_to_cpp


class CppMicroGemm:
    # TODO(jgong5): support constant shapes and lds as template args.
    DECLARE_KERNEL = r"""
template <bool accum>
inline void {{kernel_name}}(
    const {{input_t}}* __restrict__ A,
    const {{input_t}}* __restrict__ B,
    {{output_t}}* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
)
"""

    def __init__(
        self,
        name,
        input_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
    ):
        self.name = name
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.register_blocking = register_blocking
        self.alpha = alpha

    def get_common_options(self):
        return {
            "kernel_name": self.name,
            "input_t": DTYPE_TO_CPP[self.input_dtype],
            "output_t": DTYPE_TO_CPP[self.output_dtype],
            "compute_t": DTYPE_TO_CPP[self.compute_dtype],
            "alpha": self.alpha,
        }

    def get_kernel_declaration(self):
        options = self.get_common_options()
        return KernelTemplate._template_from_string(self.DECLARE_KERNEL).render(options)

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        raise NotImplementedError

    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: ir.Buffer,
        B: ir.Buffer,
        C: ir.Buffer,
        accum: bool,
    ) -> str:
        A_ptr = f"&({kernel.index(A, [0, 0])})"
        B_ptr = f"&({kernel.index(B, [0, 0])})"
        C_ptr = f"&({kernel.index(C, [0, 0])})"
        M = kernel.size(C, 0)
        N = kernel.size(C, 1)
        K = kernel.size(A, 1)
        lda = kernel.stride(A, 0)
        ldb = kernel.stride(B, 0)
        ldc = kernel.stride(C, 0)
        res = IndentedBuffer()
        res.writeline(f"{self.name}<{value_to_cpp(accum, 'bool')}>(")
        with res.indent():
            res.writeline(f"{A_ptr},")
            res.writeline(f"{B_ptr},")
            res.writeline(f"{C_ptr},")
            res.writeline(f"{M},")
            res.writeline(f"{N},")
            res.writeline(f"{K},")
            res.writeline(f"{lda},")
            res.writeline(f"{ldb},")
            res.writeline(f"{ldc}")
        res.writeline(");")
        return res.getvalue()


CppMicroGemmConfig = namedtuple(
    "CppMicroGemmConfig",
    [
        "input_dtype",
        "output_dtype",
        "compute_dtype",
        "vec_isa_cls",
        "register_blocking",
    ],
)

micro_gemm_configs: Dict[Type[CppMicroGemm], List[CppMicroGemmConfig]] = {}


def register_micro_gemm(*configs):
    def inner(cls):
        assert (
            cls not in micro_gemm_configs
        ), f"Duplicate micro_gemm registration for {cls}"
        assert len(configs) > 0, f"No micro_gemm configs provided for {cls}"
        micro_gemm_configs[cls] = list(configs)
        return cls

    return inner


class CppMicroGemmRef(CppMicroGemm):
    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            {{compute_t}} result = accum ? C[m * ldc + n] : 0;
            for (int64_t k = 0; k < K; ++k) {
                result += ({{compute_t}})A[m * lda + k] * ({{compute_t}})B[k * ldb + n] * {{alpha}};
            }
            C[m * ldc + n] = result;
        }
    }
}
"""

    def __init__(self, name, input_dtype, output_dtype, compute_dtype, alpha):
        super().__init__(
            name, input_dtype, output_dtype, compute_dtype, GemmBlocking(1, 1, 1), alpha
        )

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            **self.get_common_options(),
        }
        return KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(options)


@register_micro_gemm(
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX512, GemmBlocking(8, 48, 1)
    ),
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX512, GemmBlocking(8, 32, 1)
    ),
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX512, GemmBlocking(16, 16, 1)
    ),
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX2, GemmBlocking(4, 24, 1)
    ),
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX2, GemmBlocking(4, 16, 1)
    ),
    CppMicroGemmConfig(
        torch.float32, torch.float32, torch.float32, VecAVX2, GemmBlocking(8, 8, 1)
    ),
)
class CppMicroGemmFP32AVX(CppMicroGemm):
    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    TORCH_CHECK(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    TORCH_CHECK(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            if (block_m == {{block_m}}) {
                {{kernel_name}}_kernel<{{block_m}}, {{block_n}}, accum>(
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc
                );
            } else {
                switch (block_m) {
                {%- for b in range(block_m - 1, 0, -1) %}
                case {{b}}:
                    {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                {%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: ", block_m);
                }
            }
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""
template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum>
inline void {{kernel_name}}_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    using Vectorized = at::vec::Vectorized<float>;
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;

    Vectorized va;
    at::vec::VectorizedN<float, COLS> vb;
    at::vec::VectorizedN<float, ROWS*COLS> vc;

    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
        } else {
            vc[i] = Vectorized(0.0f);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);

    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;

        if constexpr (col == 0) {
            {%- if alpha != 1 %}
            va = Vectorized(A[row * lda + k] * {{alpha}});
            {%- else %}
            va = Vectorized(A[row * lda + k]);
            {%- endif %}
        }

        if constexpr (row == 0) {
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
        }

        constexpr int idx = row * COLS + col;
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    {{kernel.unroll_pragma(4)}}
    for (int k = 0; k < K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        vc[i].store(C + row * ldc + col * VLEN);
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
}
"""

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            **self.get_common_options(),
        }
        result = KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
            options
        )
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        return result


def create_micro_gemm(
    name,
    m,
    n,
    k,
    input_dtype,
    output_dtype=None,
    compute_dtype=None,
    alpha=1,
    num_threads=-1,
    use_ref=False,
) -> Optional[CppMicroGemm]:
    def create_from_config(cls, config: CppMicroGemmConfig):
        return cls(
            name,
            config.input_dtype,
            config.output_dtype,
            config.compute_dtype,
            config.register_blocking,
            alpha,
        )

    assert isinstance(n, int) or n.is_number, n
    assert isinstance(k, int) or k.is_number, k
    m = V.graph.sizevars.size_hint(m) if isinstance(m, sympy.Expr) else m
    assert isinstance(m, int), m
    if output_dtype is None:
        output_dtype = input_dtype
    if compute_dtype is None:
        compute_dtype = input_dtype
    if num_threads < 0:
        num_threads = parallel_num_threads()
    vec_isa = pick_vec_isa()
    matched_configs = []
    for cls, configs in micro_gemm_configs.items():
        for config in configs:
            if not isinstance(vec_isa, config.vec_isa_cls):
                continue
            if (
                config.input_dtype == input_dtype
                and config.output_dtype == output_dtype
                and config.compute_dtype == compute_dtype
            ):
                block_m, block_n, block_k = config.register_blocking
                # TODO(jgong5): support n % n_block_size != 0
                if n % block_n != 0:
                    continue
                # Criteria on the ranking of configurations
                # 1. Dividable by block sizes (block_m, block_k)
                # 2. Number of mxn blocks is large enough to occupy all the threads
                # 3. Register blocks are larger
                dividable_score = 0
                if k % block_k == 0:
                    dividable_score += 1
                if m % block_m == 0:
                    dividable_score += 1
                occupancy_score = 0
                n_blocks = n // block_n
                total_mxn_blocks = n // block_n * ((m + block_m - 1) // block_m)
                if n_blocks >= num_threads:
                    occupancy_score += 1
                if total_mxn_blocks >= num_threads:
                    occupancy_score += 1
                matched_configs.append(
                    (
                        (dividable_score, occupancy_score, block_m * block_n * block_k),
                        cls,
                        config,
                    )
                )
    if len(matched_configs) == 0:
        if use_ref:
            return CppMicroGemmRef(
                name, input_dtype, output_dtype, compute_dtype, alpha
            )
        else:
            return None
    # TODO(jgong5): allow autotuning on choices of configs
    return create_from_config(*max(matched_configs, key=lambda x: x[0])[1:])
