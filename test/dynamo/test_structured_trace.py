# Owner(s): ["module: dynamo"]
import contextlib
import functools
import logging
import json
import os
import re
import io
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.distributed as dist
from torch._dynamo.testing import skipIfNotPy311

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.testing._internal.common_utils import (
    find_free_port,
    munge_exc,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


class StructuredTraceTestingFormatter:
    def format(self, record):
        metadata = dict(record.metadata)

        # Stub out values that are not stable across runs
        # TODO: Check that these match schema
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "compile_stack" in metadata:
            metadata["compile_stack"] = "STACK"

        return json.dumps(metadata)


trace_log = logging.getLogger("torch.__trace")


class StructuredTraceTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.buffer = io.StringIO()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        trace_log.addHandler(self.handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.setLevel(self.old_level)

    @requires_cuda
    def test_schedule(self):
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    @requires_cuda
    def test_cudagraphs(self):
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    def test_recompiles(self):
        def fn(x, y):
            return torch.add(x, y)

        fn_opt = torch._dynamo.optimize("inductor")(fn)
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1000, 1000], "l_y_": [1000, 1000], "add": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1000, 1000], "add": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    def test_example_fn(self):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(torch.ones(1000, 1000))
        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000], "ones_1": [1000, 1000], "output_1": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    def test_dynamo_error(self):
        try:
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""")  # noqa: B950

    def test_inductor_error(self):
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError()

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        with unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries):
            try:
                fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
                fn_opt(*ARGS)
            except Exception:
                pass

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": [1000, 1000], "output": [1000, 1000]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"aot_joint_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    @requires_distributed()
    @requires_cuda
    def test_ddp_graphs(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.layers(x)

        # TODO: this isn't safely bracketed, will leak
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        ddp_model = torch._dynamo.optimize("inductor")(
            DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        )

        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        dist.destroy_process_group()

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compile_stack": "STACK", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1024, 1024], "l__self___layers_0": [1024, 1024], "l__self___layers_1": [1024, 1024]}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"optimize_ddp_split_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_0"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_1"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    def test_graph_breaks(self):
        @torch._dynamo.optimize("inductor")
        def fn(x):
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compile_stack": "STACK", "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1], "add": [1]}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"aot_forward_graph": true, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": true, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": true, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
""")  # noqa: B950

    # TODO: bring in the trace_source tests once we start emitting bytecode

    def test_graph_sizes_dynamic(self):
        def fn(a, b):
            return a @ b

        fn_opt = torch._dynamo.optimize("eager", dynamic=False)(fn)
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        fn_opt2 = torch._dynamo.optimize("eager", dynamic=True)(fn)
        fn_opt2(torch.randn(5, 10), torch.randn(10, 15))

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": [10, 20], "l_b_": [20, 30], "matmul": [10, 30]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_a_": ["s0", "s1"], "l_b_": ["s1", "s3"], "matmul": ["s0", "s3"]}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""")  # noqa: B950


    def test_guards_recompiles(self):
        def fn(x, ys, zs):
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch._dynamo.optimize("eager")(fn)
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        self.assertExpectedInline(self.buffer.getvalue(), """\
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1], "x": [1]}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"compile_stack": "STACK", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": true, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_output_graph_sizes": {"l_x_": [1], "x": [1]}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""")  # noqa: B950


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
