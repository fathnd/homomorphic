# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

import io
import unittest
from typing import Dict, Optional, Type

import onnx

import torch
from torch import Tensor
from torch.onnx import OperatorExportTypes, symbolic_helper, utils
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    suppress_warnings,
)


class TestOptionalOutput(TestCase):
    # TODO: Move these tests to test_pytorch_onnx_onnxruntime once
    # ONNX Runtime 1.11 is released and supports opset 16.

    class IfNoneInput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            if x.size(0) > 1:
                y = x
            return y

    class IfNoneOutput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            if x.size(0) > 1:
                y = None
            return y

    class LoopNoneInput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            for _ in range(x.size(0)):
                y = x
            return y

    class LoopNoneOutput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            for _ in range(x.size(0)):
                y = None
            return y

    @parametrize(
        "module_class",
        (IfNoneInput, IfNoneOutput, LoopNoneInput, LoopNoneOutput),
        name_fn=lambda module_class: module_class.__name__,
    )
    @parametrize("x_size", (0, 1), name_fn=lambda x_size: str(x_size))
    def test_optional_output(self, module_class: Type[torch.nn.Module], x_size: int):
        # Need scripting to preserve control flow for this test to be
        # meaningful.
        model = torch.jit.script(module_class())
        f = io.BytesIO()
        x = torch.ones(x_size)
        dynamic_axis_name = "condition"
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=15,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )
        exported = onnx.load_from_string(f.getvalue())
        expected_elem_type = symbolic_helper.scalar_type_to_onnx[
            symbolic_helper.scalar_type_to_pytorch_type.index(x.dtype)
        ].value
        expected_output_type = onnx.helper.make_optional_type_proto(
            onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,))
        )
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        for node in exported.graph.node:
            # Both branches output types should match.
            if node.op_type == "If":
                for attr in node.attribute:
                    if attr.name in ("then_branch", "else_branch"):
                        self.assertEqual(expected_output_type, attr.g.output[0].type)

    def test_uninitialized_optional(self):
        class Module(torch.nn.Module):
            def forward(self, y: Optional[Tensor]) -> Optional[Tensor]:
                if y is not None:
                    if y.shape[1] < 5:
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            return y
                return y

        y = torch.ones((3, 4), dtype=torch.int)
        torch.onnx.export(
            torch.jit.script(Module()),
            y,
            io.BytesIO(),
            opset_version=15,
            dynamic_axes={"y": {0: "y0", 1: "y1"}},
            input_names=["y"],
        )


class TestONNXExport(TestCase):
    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        f = io.BytesIO()
        torch.onnx._export(AddmmModel(), x, f, verbose=False)

    def test_onnx_transpose_incomplete_tensor_type(self):
        # Smoke test to get us into the state where we are attempting to export
        # a transpose op, where the input is a TensorType without size information.
        # This would previously not work, since we would
        # take the size of the input and use the length of its sizes as the
        # number of dimensions in the permutation.
        class Foo(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.contiguous().transpose(0, 1).sum()

        class TraceMe(torch.nn.Module):
            def __init__(self):
                super(TraceMe, self).__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        tm = TraceMe()
        tm = torch.jit.trace(tm, torch.rand(3, 4))
        f = io.BytesIO()
        torch.onnx._export(tm, (torch.rand(3, 4),), f)

    def test_export_tensoroption_to(self):
        def foo(x):
            return x[0].clone().detach().cpu() + x

        traced = torch.jit.trace(foo, (torch.rand([2])))

        torch.onnx.export_to_pretty_string(traced, (torch.rand([2]),))

    def test_onnx_export_script_module(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                y = x - x
                return x + x

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    @suppress_warnings
    def test_onnx_export_func_with_warnings(self):
        @torch.jit.script
        def func_with_warning(inp):
            return torch.nn.functional.sigmoid(inp)  # triggers a deprecation warning

        class WarningTest(torch.nn.Module):
            def __init__(self):
                super(WarningTest, self).__init__()

            def forward(self, x):
                return func_with_warning(x)

        # no exception
        torch.onnx.export_to_pretty_string(
            WarningTest(), torch.randn(42), verbose=False
        )

    def test_onnx_export_script_python_fail(self):
        class PythonModule(torch.jit.ScriptModule):
            def __init__(self):
                super(PythonModule, self).__init__()

            @torch.jit.ignore
            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = PythonModule()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        f = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "Couldn't export Python"):
            torch.onnx._export(mte, (torch.zeros(1, 2, 3),), f, verbose=False)

    def test_onnx_export_script_inline_trace(self):
        class ModuleToInline(torch.nn.Module):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = torch.jit.trace(ModuleToInline(), torch.zeros(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_inline_script(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_module_loop(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                # test if we support end to end onnx export on loop and
                # nested loops with and without loop index
                for _ in range(5):
                    for i in range(3):
                        x = x + i
                return x

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    @suppress_warnings
    def test_onnx_export_script_truediv(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                z = x.size(0) / 2
                return x + z

        mte = ModuleToExport()

        torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3, dtype=torch.float),), verbose=False
        )

    def test_onnx_export_script_non_alpha_add_sub(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                bs = x.size(0) + 1
                return bs - 1

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.rand(3, 4),), verbose=False)

    def test_onnx_export_script_module_if(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                if bool(torch.sum(x) > 0):
                    x = torch.neg(x)
                return x

        mte = ModuleToExport()
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_inline_params(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()
                self.m = torch.nn.Parameter(torch.ones(3, 3))
                self.unused = torch.nn.Parameter(torch.ones(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.m)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()
                self.param = torch.nn.Parameter(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return torch.mm(y, self.param)

        mte = ModuleToExport()
        result = mte(torch.zeros(2, 3))
        reference = torch.mm(
            torch.mm(torch.zeros(2, 3), torch.ones(3, 3)), torch.ones(3, 4)
        )
        self.assertEqual(result, reference)
        torch.onnx.export_to_pretty_string(mte, (torch.ones(2, 3),), verbose=False)

    def test_onnx_export_speculate(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self, m):
                super(Foo, self).__init__()
                self.m = m

            @torch.jit.script_method
            def forward(self, x):
                x += x
                # because we are testing if we emit `if` statement correctly
                # we cannot use `True` as the condition. Constant prop
                # would remove the `if` statements.
                c = torch.sum(x) > 4
                if bool(c):
                    if bool(c):
                        y = self.m(x)
                    else:
                        y = self.m(x)
                else:
                    y = self.m(x)
                return y

        linear = torch.jit.trace(
            torch.nn.Linear(10, 20).float(), torch.zeros(1, 10, dtype=torch.float)
        )

        @torch.jit.script
        def transpose(x):
            return x.t()

        f1 = Foo(transpose)
        f2 = Foo(linear)

        torch.onnx.export_to_pretty_string(f1, (torch.ones(1, 10, dtype=torch.float),))
        torch.onnx.export_to_pretty_string(f2, (torch.ones(1, 10, dtype=torch.float),))

    def test_onnx_export_shape_reshape(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                import torch.onnx.operators

                x = x.repeat(5, 1, 1)
                shape = torch.onnx.operators.shape_as_tensor(x)
                reshaped = torch.onnx.operators.reshape_from_tensor_shape(x, shape)
                return reshaped

        foo = torch.jit.trace(Foo(), torch.zeros(1, 2, 3))
        torch.onnx.export_to_pretty_string(foo, (torch.zeros(1, 2, 3)))

    def test_listconstruct_erasure(self):
        class FooMod(torch.nn.Module):
            def forward(self, x):
                mask = x < 0.0
                return x[mask]

        torch.onnx.export_to_pretty_string(
            FooMod(),
            (torch.rand(3, 4),),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    def test_export_dynamic_slice(self):
        class DynamicSliceExportMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                retval = x[0]
                for i in range(x.size(1)):
                    retval += torch.sum(x[0:i], dim=0)
                return retval

        mod = DynamicSliceExportMod()

        input = torch.rand(3, 4, 5)

        torch.onnx.export_to_pretty_string(
            DynamicSliceExportMod(), (input,), opset_version=10
        )

    def test_export_dict(self):
        class DictModule(torch.nn.Module):
            def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"test_key_out": x_in}

        x_in = torch.tensor(1)
        mod = DictModule()
        mod.train(False)

        torch.onnx.export_to_pretty_string(mod, (x_in,))

        with self.assertRaisesRegex(RuntimeError, r"DictConstruct.+is not supported."):
            torch.onnx.export_to_pretty_string(torch.jit.script(mod), (x_in,))

    def test_source_range_propagation(self):
        class ExpandingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Will be expanded during ONNX export
                self.ln = torch.nn.LayerNorm([1])

            def forward(self, input):
                return self.ln(input)

        mod = ExpandingModule()

        graph, _, _ = utils._model_to_graph(
            mod, (torch.zeros(1),), operator_export_type=OperatorExportTypes.ONNX
        )

        # Ensure that every node in the graph has a valid source range
        for node in graph.nodes():
            self.assertTrue(node.sourceRange())


instantiate_parametrized_tests(TestOptionalOutput)

if __name__ == "__main__":
    unittest.main()
