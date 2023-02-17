# Owner(s): ["module: dynamo"]
import functools
import unittest
from unittest.mock import patch
import torch
from torch._C import FileCheck
from torch._dispatch.python import enable_python_dispatcher
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch._dynamo.testing import CompileCounter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    DynamoDistributedMultiProcTestCase,
    _dynamo_dist_per_rank_init,
    requires_nccl,
    skip_if_lt_x_gpu
)
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.utils import has_triton, run_and_get_triton_code
import torch._dynamo.logging

# LOL if you don't remember to import this, then the op isn't registered and it hits
# the no-op C++ kernel that i am forced to implement despite not using it
import torch.distributed._functional_collectives


@requires_nccl()
class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """
    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allreduce_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def matmul_cat_col(a, b, c, d, e, f, *, tag, ranks, group_size):
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.aten.all_reduce(z, "sum", tag, ranks, group_size)
            g = torch.matmul(e, f)
            ar = torch.ops.aten.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out, )

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):

            matmul_cat_col = functools.partial(
                matmul_cat_col,
                **self.get_world_trs(),
            )
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 6

            # non-ideally, i seem to need to enable this at user level in order to construct a torchdispatch subclass
            # inside py registered collective ops
            with enable_python_dispatcher():
                eager_out = matmul_cat_col(*inputs)
                compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
                inductor_out = compiled_matmul_cat_col(*inputs)
                assert same(eager_out, inductor_out, tol=0.001)


@requires_nccl()
class TestCollectivesInductor(DynamoDistributedSingleProcTestCase):
    """
    Prefer single-proc test runner for basic tests as it is easier to work with.
    """
    def get_world_trs(self, world_size=1):
        return {
            "tag": "",
            "ranks": list(range(world_size)),
            "group_size": world_size,
        }

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_inductor_single_op(self):
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.aten.all_reduce(inp, "sum", tag, ranks, group_size)
            ar = torch.ops.aten.wait_tensor(ar)
            return ar

        inputs = torch.ones(4, 4, device="cuda")

        with enable_python_dispatcher():
            compiled = torch.compile(func)
            out = compiled(inputs, **self.get_world_trs())
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck() \
                .check("buf0 = empty_strided") \
                .check("buf0.copy_(arg0_1)") \
                .check("buf0_work = dist.all_reduce(buf0") \
                .check("buf0_work.wait()") \
                .check("return (buf1, )") \
                .run(code)
            correct = func(inputs, **self.get_world_trs())
            assert same(out, correct)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_inductor_steal_buffer(self):
        """
        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate
        that isn't going to be used again
        """
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = torch.ops.aten.all_reduce(x, "sum", tag, ranks, group_size)
            ar = torch.ops.aten.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar, other

        inputs = torch.ones(4, 4, device="cuda")

        with enable_python_dispatcher():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck() \
                .check("buf1 = buf0; del buf0  # reuse") \
                .check_not("buf1.copy_(") \
                .check("buf1_work = dist.all_reduce(buf1") \
                .check("buf1_work.wait()") \
                .check("buf2 = buf1") \
                .check("buf3 = empty_strided") \
                .check("return (buf2, buf3") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            assert same(out, correct)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_inductor_doesnt_mutate_shared(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = torch.ops.aten.all_reduce(x, "sum", tag, ranks, group_size)
            y = x + 2
            ar = torch.ops.aten.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar, y, other

        inputs = torch.ones(4, 4, device="cuda")

        with enable_python_dispatcher():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            FileCheck() \
                .check("buf0 = empty_strided(") \
                .check("buf2 = empty_strided") \
                .check("triton__0.run(arg0_1, buf0, buf2") \
                .check_not("copy_(") \
                .check("buf1 = buf0; del buf0  # reuse") \
                .check("buf1_work = dist.all_reduce(buf1") \
                .check("buf1_work.wait()") \
                .check("buf3 = buf1") \
                .check("return (buf3, buf2, buf4") \
                .run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            assert same(out, correct)

    def test_dynamo_trace_allreduce(self):
        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.aten.all_reduce(inp, "sum", tag, ranks, group_size)
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        with enable_python_dispatcher():
            compiled = torch.compile(func, backend=counter)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            assert counter.frame_count == 1
            assert counter.op_count == 1
            assert same(out, correct)

    def test_backwards(self):
        """
        It's probably not that common to need backwards support for collectives.

        However, I wanted to at least see if it was possible to support it as a design goal.
        """
        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.aten.all_reduce(inp, "sum", tag, ranks, group_size)
            return ar

        input = torch.ones(4, 4, device="cuda", requires_grad=True)
        with enable_python_dispatcher():
            # TODO implement backwards
            with self.assertRaisesRegex(RuntimeError, "derivative for aten::all_reduce is not implemented"):
                compiled = torch.compile(func, backend="aot_eager")  # inductor bug with single-op allreduce graph
                out = compiled(input, **self.get_world_trs())
                out.sum().backward()

                correct_input = input.clone().detach().requires_grad_()
                correct = func(correct_input, **self.get_world_trs())
                correct.sum().backward()
                assert same(out, correct)
                assert same(input.grad, correct_input.grad)

    def test_meta(self):
        x = torch.rand((2, 3, 4), device="meta")
        out = torch.ops.aten.all_reduce(x, "sum", **self.get_world_trs())
        assert x.size() == out.size()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
