import copy
import enum
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from ... import ir
from ...config import cuda as inductor_cuda_config
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    FixedLayout,
    IRNode,
    Layout,
    ReinterpretView,
)
from ..common import IndentedBuffer

from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate

log = logging.getLogger(__name__)

# Jinja template for GEMM Kernel, used by the CUTLASSGemmTemplate class below.
GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
{{kernel_call_signature}} {
  try {
  int64_t B = {{kernel.size(Y, 0, -3, default_value=1)}};
  int64_t M = {{kernel.size(X, -2)}};
  int64_t K = {{kernel.size(X, -1)}};
  int64_t N = {{kernel.size(W, -1)}};
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  {{kernel.check_not_null(X)}}
  {{kernel.check_not_null(W)}}
  {{kernel.check_not_null(Bias)}}
  {{kernel.check_not_null(Y)}}
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
#endif
#ifdef CUTLASS_DEBUG_TRACE_LEVEL
#if CUTLASS_DEBUG_TRACE_LEVEL == 1
  {
    // Print the maximum number of active blocks per SM for the kernel if CUTLASS_DEBUG_TRACE_LEVEL == 1
    // we don't need a print statement, it's happening inside the function.
    gemm_op.maximum_active_blocks();
  }
#endif
#endif
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }
  return 0;
}
}
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments, used by the CUTLASSGemmTemplate class below.
GEMM_ARGS_CUTLASS_3X = r"""
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>({{M}}),
      static_cast<coord_t>({{N}}),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // ElementA const* ptr_A
      {
        {{template.cute_int(kernel.stride(X, -2), "stride_x0")}},
        {{template.cute_int(kernel.stride(X, -1), "stride_x1")}},
        {{template.cute_int(kernel.stride(X, -3), "batch_stride_x")}}
      },  // StrideA dA
      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B
      {
        {{template.cute_int(kernel.stride(W, -1), "stride_w1")}},
        {{template.cute_int(kernel.stride(W, -2), "stride_w0")}},
        {{template.cute_int(kernel.stride(W, -3), "batch_stride_w")}}
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}},
    hw_info
  };
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments if epilogue fusion is applied,
# used by the CUTLASSGemmTemplate class below.
GEMM_ARGS_CUTLASS_3X_EPILOGUE = r"""
    // see https://tinyurl.com/4rk89z48
    {
      {{epilogue_args}},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )
      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C
      {
        {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}},
        {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}},
        {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}
      },  // StrideC dC
      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D
      {
        {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}},
        {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}},
        {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}
      },  // StrideD dD
    },  // EpilogueArguments epilogue
"""

# Additional includes which are neccessary if the standalone test / debug runner is generated as wel
GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES = r"""
#ifdef GENERATE_STANDALONE_RUNNER
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include <iostream>
#endif
"""

# Jinja template for the standalone runner that may be generated as part of the code.
GEMM_STANDALONE_RUNNER_TEMPLATE = r"""
#ifdef GENERATE_STANDALONE_RUNNER
/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed, float max=1.0, float min=-1.0) {
  if (block.size()<=0) return false;
  Element scope_max(static_cast<Element>(max)), scope_min(static_cast<Element>(min));
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

extern "C" int run_standalone(uint64_t seed, int repetitions) {
    std::cout << "Starting GEMM Standalone test run with seed " << seed << std::endl;
    size_t workspace_size = 0;
    size_t* workspace_size_ptr = &workspace_size;

    using ElementA = {{kernel.cutlass_dtype(X)}};
    using ElementB = {{kernel.cutlass_dtype(W)}};
    using ElementC = {{kernel.cutlass_dtype(Bias, default_dtype='uint8_t')}}; // may not be void
    using ElementD = {{kernel.cutlass_dtype(Y)}};

    cutlass::DeviceAllocation<ElementA> X_data({{kernel.max_valid_index(X)+1}});
    initialize_block(X_data, seed++);
    cutlass::DeviceAllocation<ElementB> W_data({{kernel.max_valid_index(W)+1}});
    initialize_block(W_data, seed++);
    cutlass::DeviceAllocation<ElementC> Bias_data({{kernel.max_valid_index(Bias)+1}});
    initialize_block(Bias_data, seed++);
    cutlass::DeviceAllocation<ElementD> Y_data({{kernel.max_valid_index(Y)+1}});

    cutlass::DeviceAllocation<uint8_t> workspace_data;
    // Call once with workspace_size_ptr set to get workspace size

    std::cout << "Calling once to get workspace size" << std::endl;
    {{test_call_statement}};
    // Allocate workspace if neccessary
    if (workspace_size > 0) {
        workspace_data.reset(workspace_size);
        std::cout << "Allocated workspace size of " << workspace_size << " bytes" << std::endl;
    }
    std::cout << "Calling Kernel as {{test_call_statement}};" << std::endl;
    workspace_size_ptr = nullptr;
    for (int i=0; i<repetitions; i++) {
        {{test_call_statement}};
    }
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
    return 0;
}

int main(int argc, char** argv) {
    // warmup
    run_standalone(1, 2);
    // repeat
    return run_standalone(2, 10);
}

#endif
"""  # noqa: B950


class CUTLASSGemmTemplate(CUTLASSTemplate):
    """
    CUTLASS GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        """
        Args:
            input_nodes (List[Buffer]): List of input nodes of the GEMM kernel.
            layout (Layout): Layout type of the resulting output node.
            alpha (float): The scaling factor for the product of the inputs in the GEMM operation.
            beta (float): The scaling factor applied to the output matrix.
            input_reorder (Optional[List[int]]): Specifies the reordering of the input nodes. If not provided,
                            no reordering is performed. Defaults to None.
        """
        super().__init__("cutlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta
        assert len(input_nodes) == 2 or len(input_nodes) == 3
        assert self._are_inputs_layout_compatible(
            [node.get_layout() for node in input_nodes]
        )

    def _are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for General Matrix Multiply (GEMM).

        This function checks compatibility of A, B, and possibly C operand layouts for
        a General Matrix Multiply (GEMM) operation, expressed as 'alpha * matmul(A, B) + beta * C'.
        It verifies requirements such as matching data types, minimum rank, and suitability
        for broadcasting, as defined by PyTorch operations like `torch.matmul`, `torch.aten.mm`,
        `addmm`, `bmm`, `baddbmm`, etc.

        Args:
            layouts (List[Layout]): List containing 2 or 3 Layout objects representing
                                    the input matrices A, B, and possibly C.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        assert len(layouts) == 2 or len(layouts) == 3
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) < 1:
            return False
        if len(B_layout.size) < 1:
            return False
        A_size = [int(i) for i in A_layout.size]
        B_size = [int(i) for i in B_layout.size]
        if len(A_size) < 2:
            A_size.insert(0, 1)
        if len(B_size) < 2:
            A_size.insert(1, 1)
        # Are batch dims broadcastable?
        while len(A_size) < len(B_size):
            A_size.insert(0, 1)
        while len(B_size) < len(A_size):
            B_size.insert(0, 1)
        K = max(A_size[-1], B_size[-2])
        M = A_size[-2]
        N = B_size[-1]
        if K != A_size[-1] and A_size[-1] != 1:
            return False
        if K != B_size[-2] and B_size[-1] != 1:
            return False
        # check batch dim broadcastable
        for i in range(len(A_size) - 2):
            if A_size[i] != B_size[i] and A_size[i] != 1 and B_size[i] != 1:
                return False
        if len(layouts) == 3:
            C_layout = layouts[2]
            C_size = [int(i) for i in C_layout.size]
            while len(C_size) < len(A_size):
                C_size.insert(0, 1)
            # check batch dims
            for i in range(len(A_size) - 2):
                bd = max(A_size[i], B_size[i])
                if bd != C_size[i] and C_size[i] != 1:
                    return False
            if len(C_size) > len(A_size):
                # This may happen if the last elements of C are contiguous and
                # their multiplied size equals the last dim size of B
                if M != C_size[len(A_size) - 2] and C_size[len(A_size) - 2] != 1:
                    return False
                remaining_size = 1
                for i in range(len(A_size) - 1, len(C_size)):
                    remaining_size *= C_size[i]
                if N != remaining_size and remaining_size != 1:
                    return False
                return True
            assert len(C_size) == len(A_size)
            if M != C_size[-2] and C_size[-2] != 1:
                return False
            if N != C_size[-1] and C_size[-1] != 1:
                return False
        return True

    @staticmethod
    def add_cutlass_gemm_choices(
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[ir.IRNode],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        """
        Adds Cutlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Cutlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """

        cutlass_template = CUTLASSGemmTemplate(
            input_nodes,  # type: ignore[arg-type]
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        ops = cutlass_template.gen_ops()
        for op in ops:
            cutlass_template.maybe_append_choice(
                choices,
                op=op,
            )
        if len(ops) == 0:
            input_layouts = [node.get_layout() for node in input_nodes]
            input_strides = [node.get_stride() for node in input_nodes]
            output_layout = layout
            warning_msg = f"No suitable Cutlass GEMM configs found, fallbacks used ( {len(ops)=}, {output_layout=}, {input_layouts=}, {input_strides=} )"  # noqa: B950
            log.warning(warning_msg)
        log.debug(
            "Added %d Cutlass gemm configs.",
            len(ops),
        )

    def header(self) -> IndentedBuffer:
        """
        Returns a buffer containing CUDA C++ code for the header section of the CUTLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated CUDA C++ header code.
        """
        res = super().header()
        res.splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/epilogue/collective/default_epilogue.hpp"
                #include "cutlass/epilogue/thread/linear_combination.h"
                #include "cutlass/epilogue/thread/activation.h"
                #include "cutlass/gemm/dispatch_policy.hpp"
                #include "cutlass/gemm/kernel/tile_scheduler.hpp"
                #include "cutlass/util/distribution.h"
                #include "cutlass/util/packed_stride.hpp"
                #include "cutlass/util/tensor_view_io.h"
            """
        )
        if inductor_cuda_config.generate_test_runner:
            res.splice(GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES)
        return res

    @staticmethod
    def cutlass_layout(torch_layout: ir.Layout) -> "Optional[cutlass_lib.LayoutType]":  # type: ignore[name-defined]  # noqa: F821
        """
        Converts an ir.Layout instance into the corresponding cutlass_library.LayoutType enum value
        (RowMajor, ColumnMajor, or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            cutlass_lib.LayoutType: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if torch_layout.stride[-1] == 1:
            return cutlass_lib.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_lib.LayoutType":  # type: ignore[name-defined]  # noqa: F821
        """Helper method: Flips a given cutlass layout (cutlass_lib.LayoutType) from RowMajor
        to ColumnMajor or vice versa"""
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    def layout_match(
        torch_layout: ir.Layout,
        cutlass_layout: "cutlass_lib.LayoutType",  # type: ignore[name-defined] # noqa: F821
    ) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Cutlass layout"""
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        """
        Helper method to update the alignment of a given CUTLASS GEMM op operand's element.

        This method modifies the alignment of the given Cutlass GEMM op operand's element to match the
        layout of the corresponding ir.Buffer node.

        Args:
            torch_layout: The layout of the corresponding ir.Buffer node.
            op_element: The Cutlass GEMM op operand's element whose alignment is to be updated.

        Returns:
            bool: True if the alignment was successfully updated, False otherwise.
        """
        alignment = cutlass_utils.get_max_alignment(torch_layout)
        cuda_arch = cutlass_utils.get_cuda_arch()
        if cuda_arch and int(cuda_arch) >= 90 and alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def has_tma_epilogue(  # noqa: F821 # type: ignore[arg-type,name-defined]
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined,arg-type] # noqa: F821
    ) -> bool:  # type: ignore[name-defined]
        """Helper method: Determine whether a given Cutlass GEMM op has a TMA Epilogue"""
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result

    def define_gemm_instance(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> Tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            Tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
        if not hasattr(op, "epilogue_functor") or not isinstance(
            op.epilogue_functor, enum.Enum
        ):
            op = copy.deepcopy(op)
            op.epilogue_functor = cutlass_lib.EpilogueFunctor.LinearCombination
        op_def = emitter.emit(op)
        pattern = re.compile(r"\s*struct\s(.*?)\s:")
        decl = [line for line in op_def.split("\n") if "struct " in line][-1]

        match = pattern.match(decl)
        if match is None:
            raise RuntimeError("Invalid Gemm config: \n" + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op_def += f"\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n"
            op_type = f"{op_type}_device_type"
        return op_def, op_type

    @staticmethod
    def should_swap_XW(
        bias: IRNode,
    ) -> bool:
        """
        Helper method to determine whether we should do an explicit transpose by switching the order of the
        matmul operands. This might be neccessary when we can't otherwise arrive at the right memory
        layout for the given Bias operand.

        Note: This method is a workaround for CUDA Errors that seemingly non-deterministically
        occurred in practice in some CUTLASS GEMM Kernels with Linear epilogues that have a bias term.
        it might make sense to check on newer Cutlass releases whether it makes sense to keep
        returning True in certain cases or whether it becomes unneccessary.
        """
        # If bias is row major, swap all M and N dimensions
        if (
            bias is not None
            and len(bias.get_stride()) >= 2
            and bias.get_stride()[-1] in (0, 1)
        ):
            log.debug("GEMM Layout swapped X and W -> explicit transpose")
            return True
        return False

    @staticmethod
    def swap_XW(
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Swap operands X and W (aka operans A and B) of the GEMM operation. This
        requires transposing the operands, which is done by swapping the strides.
        Note that we don't change the apparent external layout, just the operand layout.
        this is intentional.
        """
        new_op = copy.deepcopy(op)
        new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        new_op.A, new_op.B = new_op.B, new_op.A
        new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
        return new_op

    def fix_op_layout(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined] # noqa: F821
        X: Buffer,
        W: Buffer,
        Bias: Optional[Buffer],
        Y: Union[Buffer, ReinterpretView],
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        # This is a workaround to deal with cases where the input layouts have changed
        # between autotuning and rendering. This happens if the inputs layout
        # are FlexibleLayout instances. In this case, we need to update the
        # op's input layouts. It is a hack, because now the op
        # we benchmarked is not the same as the op we render,
        # but there is no simple way to fix this in the autotuner, since that would
        # potentially disable other optimizations.
        a_layout = X.get_layout()
        b_layout = W.get_layout()
        c_layout = Bias.get_layout() if Bias is not None else None

        d_layout = copy.deepcopy(Y.get_layout())
        match_list = [
            CUTLASSGemmTemplate.layout_match(buf.get_layout(), op_layout)
            for buf, op_layout in zip(
                (X, W, Bias, Y),
                (op.A.layout, op.B.layout, op.C.layout, op.D.layout),
            )
            if buf is not None
        ]
        all_match = all(match_list)
        if all_match:
            return op
        log.warning(
            f"Cutlass GEMM Layout change: Input and/or output layouts have changed between autotuning/retuning and call to render on {self}. Applying workaround. This can lead to suboptimal performance. Match List: {match_list}"  # noqa: G004, B950
        )
        new_op = copy.deepcopy(op)

        if a_layout is not None:
            new_op.A.layout = CUTLASSGemmTemplate.cutlass_layout(a_layout)
        if b_layout is not None:
            new_op.B.layout = CUTLASSGemmTemplate.cutlass_layout(b_layout)
        if c_layout is not None:
            new_op.C.layout = CUTLASSGemmTemplate.cutlass_layout(c_layout)
            new_op.C.element = cutlass_utils.torch_dtype_to_cutlass_type(c_layout.dtype)
        if d_layout is not None:
            new_op.D.layout = CUTLASSGemmTemplate.cutlass_layout(d_layout)
        return new_op

    def filter_op(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> "cutlass_library.gemm_op.GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """
        Helper method:

        Determines whether a given Cutlass GEMM op definition is suitable for the current
        input / output of the operation that this template is supposed to implement.

        Takes memory layout, dtype and support for EVT operations into account,
        and filters potentially problematic ops.

        Returns None if the op is not suitable, otherwise returns the op to be used, which might
        have been mutated.
        """

        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib

        # Skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.OpcodeClass.Simt
        ):
            return None

        # Only keep GemmUniversal kernels
        if op.gemm_kind not in {
            cutlass_lib.GemmKind.Universal3x,
        }:
            return None
        # Filter ops by dtypes.
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype(
            [X.get_dtype(), W.get_dtype()],
        )
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element)
            and cutlass_utils.dtype_match(W.get_dtype(), op.B.element)
            and cutlass_utils.dtype_match(
                self.output_node.get_layout().dtype, op.C.element
            )
            and cutlass_utils.dtype_match(
                accumulator_torch_dtype, op.accumulator_type()
            )
        ):
            return None

        # Filter ops by input layouts.
        if not (
            self.layout_match(X.get_layout(), op.A.layout)
            and self.layout_match(W.get_layout(), op.B.layout)
        ):
            return None

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout.
        op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())

        # Filter ops by alignments and set alignments.
        if not (
            self.set_alignment(X.get_layout(), op.A)
            and self.set_alignment(W.get_layout(), op.B)
            and self.set_alignment(self.output_node.get_layout(), op.D)
        ):
            return None

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type()
        if inductor_cuda_config.cutlass_op_allowlist_regex is not None:
            if not re.search(
                inductor_cuda_config.cutlass_op_allowlist_regex, op.configuration_name()
            ):
                return None
        if inductor_cuda_config.cutlass_op_denylist_regex is not None:
            if re.search(
                inductor_cuda_config.cutlass_op_denylist_regex, op.configuration_name()
            ):
                return None
        # Set bias layout and alignment.
        if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
                if bias_layout != op.D.layout:
                    # For cutlass2, bias and output layout must match
                    return None
            else:
                op.C.layout = bias_layout
            if not self.set_alignment(Bias.get_layout(), op.C):
                return None
        else:
            if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
                op.C.element = cutlass_lib.DataType.void
            else:
                op.C.layout = op.D.layout
        return op

    def gen_ops(self) -> "List[cutlass_gemm_op.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        """
        Creates a list of Cutlass GemmOperation instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[cutlass_gemm_op.GemmOperation]: A list of GemmOperation instances that are compatible with the
            operation requirements of this template.
        """
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        ops = cutlass_utils.gen_ops()[cutlass_lib.OperationKind.Gemm]
        res: Dict[str, cutlass_gemm_op.GemmOperation] = dict()
        for op_dict in ops.values():
            for op_list in op_dict.values():
                for op in op_list:
                    assert isinstance(op, cutlass_gemm_op.GemmOperation)
                    filter_res = self.filter_op(op)
                    if (
                        filter_res is not None
                        and res.get(filter_res.configuration_name(), None) is None
                    ):
                        res[filter_res.configuration_name()] = filter_res
        log.debug("Got cutlass configs: total number of ops: %d, ", len(res))
        return list(res.values())[: inductor_cuda_config.cutlass_max_profiling_configs]

    def gemm_mode(self) -> str:
        """
        Returns a Cutlass GEMM mode string for the current operation, dependent on whether this op implements
        a batched GEMM or a simple GEMM without batch dimension.

        Returns:
        str: A string indicating the Cutlass GEMM mode. If the output node has more than two dimensions,
            "cutlass::gemm::GemmUniversalMode::kBatched" is returned, otherwise
            "cutlass::gemm::GemmUniversalMode::kGemm" is returned.
        """
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return "cutlass::gemm::GemmUniversalMode::kBatched"
        else:
            return "cutlass::gemm::GemmUniversalMode::kGemm"

    def render_gemm_arguments(
        self,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUDATemplateKernel,
        epilogue_args,
    ) -> str:
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUDATemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
        options = dict(
            alpha=alpha,
            beta=beta,
            X=X,
            W=W,
            Y=Y,
            Bias=Bias,
            template=self,
            kernel=kernel,
            M="M",
            N="N",
            epilogue_args=epilogue_args,
        )
        assert epilogue_template is not None

        if should_swap_xw:
            # Swap
            def clone_with_transposed_stride(node: IRNode) -> IRNode:
                old_layout = node.get_layout()
                new_stride = list(old_layout.stride)
                new_stride[-2], new_stride[-1] = new_stride[-1], new_stride[-2]
                new_layout = FixedLayout(
                    old_layout.device,
                    old_layout.dtype,
                    list(old_layout.size),
                    new_stride,
                    old_layout.offset,
                )
                return Buffer(node.get_name(), new_layout)

            new_X = clone_with_transposed_stride(X)
            new_W = clone_with_transposed_stride(W)
            new_Bias = clone_with_transposed_stride(Bias)
            new_Y = clone_with_transposed_stride(Y)
            options["X"], options["W"], options["Bias"], options["Y"] = (
                new_W,
                new_X,
                new_Bias,
                new_Y,
            )
            options["M"], options["N"] = "N", "M"

        epilogue_arguments = self._template_from_string(epilogue_template).render(
            **options
        )
        arguments = self._template_from_string(argument_template).render(
            epilogue_arguments=epilogue_arguments, **options
        )

        return arguments

    def render(  # type: ignore[override]
        self,
        kernel: CUDATemplateKernel,
        op: "cutlass_gemm_op.GemmOperation" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: Optional[CUDATemplateBuffer] = None,
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Cutlass based CUDA C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CUDATemplateKernel): The kernel to be rendered.
            op (cutlass_gemm_op.GemmOperation, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Cutlass based CUDA C++ code fragment as a string, to be used by the current
            CUDATemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """

        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib

        assert isinstance(
            op, cutlass_gemm_op.GemmOperation
        ), "op argument is required and has to be an instance of GemmOperation"

        assert len(self.input_nodes) >= 2 and self.output_node is not None
        X, W = self.input_nodes[0], self.input_nodes[1]
        assert isinstance(X.layout, FixedLayout), "X.layout is not fixed"
        assert isinstance(W.layout, FixedLayout), "W.layout is not fixed"
        Y = self.output_node
        if template_buffer_node is not None:
            Y = template_buffer_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        # to make op mutable without affecting others
        op = copy.deepcopy(op)
        if Bias is not None:
            assert Bias.get_layout().dtype == X.get_layout().dtype
            # This might have been set to void during filtering, when the assumption was still that there's no C
            # operand
            op.C.element = op.A.element

        # Define Kernel call signature
        # Important: This step also populates Kernel name to node mapping data structures,
        # which are required further below ( for example by CutlassEVTEpilogueArgumentFormatter and
        # the template renderer )
        inputs = [X, W, Bias]
        names = ["X", "W", "Bias"] + ["Y"]
        names_str = ",".join(names)
        if self.input_reorder is not None:
            input_reorder = self.input_reorder
        else:
            input_reorder = None
        kernel_call_signature = kernel.def_kernel(
            inputs=inputs, outputs=[Y], names_str=names_str, input_reorder=input_reorder  # type: ignore[arg-type]
        )
        test_call_statement = self.test_call_statement(kernel, inputs, names_str)
        # The layouts might have changed between autotuning and this call if they were FlexibleLayout
        # we need to adapt, which might lead to suboptimal performance.

        op = self.fix_op_layout(op, X, W, Bias, Y)
        epilogue_template: str = GEMM_ARGS_CUTLASS_3X_EPILOGUE
        argument_template: str = GEMM_ARGS_CUTLASS_3X
        should_swap_xw: bool = False
        epilogue_args = f"{{ElementComputeEpilogue({self.alpha}), ElementComputeEpilogue({self.beta})}}"
        if Bias is not None and self.has_tma_epilogue(op):
            if (
                op.epilogue_schedule
                != cutlass_lib.EpilogueScheduleType.EpilogueTransposed
                and self.should_swap_XW(Bias)
            ):
                # TMA epilogue requires bias vector in column major to get best perf.
                op = self.swap_XW(op)
                should_swap_xw = True

        instance_definition, instance_type = self.define_gemm_instance(op)

        options = dict(
            alpha=self.alpha,
            beta=self.beta,
            X=X,
            W=W,
            Y=Y,
            kernel_call_signature=kernel_call_signature,
            Bias=Bias,
            epilogue_template=epilogue_template,
            argument_template=argument_template,
            should_swap_xw=should_swap_xw,
            template=self,
            kernel=kernel,
            instance_definition=instance_definition,
            instance_type=instance_type,
            input_reorder=self.input_reorder,
            epilogue_args=epilogue_args,
            test_call_statement=test_call_statement,
        )
        res = self._template_from_string(GEMM_TEMPLATE).render(**options)
        if inductor_cuda_config.generate_test_runner:
            test_runner_code = self._template_from_string(
                GEMM_STANDALONE_RUNNER_TEMPLATE
            ).render(**options)
            res += "\n\n" + test_runner_code
        return res

    def test_call_statement(
        self,
        kernel,
        input_nodes,
        names_str: str = "",
    ) -> str:
        """
        Helper method to render the Cutlass CUDA C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
        _, __, arg_types = kernel.args.cpp_argdefs()
        arg_names = [name.strip() for name in names_str.strip().split(",")]
        if input_nodes[2] is None:
            del arg_names[2]
        arguments = [
            f"(({arg_type}){arg_name}_data.get())"
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        return f"{kernel.kernel_name}({', '.join(arguments)}, workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);"
