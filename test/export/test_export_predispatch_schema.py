# Owner(s): ["oncall: export"]

try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing
from torch.export._trace import _export

import copy
import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.schema_check_mode import SchemaCheckMode

test_classes = {}

class PreDispatchSchemaCheckMode(SchemaCheckMode):
    # creating this just so we have access to the offending op
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            return super().__torch_dispatch__(func, types, args=args, kwargs=kwargs)
        except RuntimeError as e:
            msg = e.args[0]
            e.args = (f"""SchemaCheckMode failed with the following error on op <{func}>, meaning
this op contains aliasing or mutations, despite claiming not to:\n\n""" + msg,)
            raise e

def _contains_higher_order_ops(graph: torch.fx.Graph):
    for node in graph.nodes:
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            return True
    return False

def _contains_fake_inputs(args, kwargs):
    for arg in pytree.tree_flatten(args)[0]:
        if isinstance(arg, FakeTensor):
            return True
    for arg in pytree.tree_flatten(kwargs)[0]:
        if isinstance(arg, torch.Tensor):
            return True
    return False

def _contains_fake_weights(model):
    for param in model.parameters():
        if isinstance(param, FakeTensor):
            return True
    for buffer in model.buffers():
        if isinstance(buffer, FakeTensor):
            return True
    return False

def mocked_schema_check_export(*args, **kwargs):
    # If user already specified strict, don't make it non-strict
    model = copy.deepcopy(args[0] if args else kwargs["mod"])
    eager_args = copy.deepcopy(args[1] if len(args) > 1 else kwargs["args"])
    eager_kwargs = copy.deepcopy(args[2] if len(args) > 2 else kwargs.get("kwargs", {}))
    ep = _export(*args, **kwargs, pre_dispatch=True).run_decompositions()
    if (
        not _contains_higher_order_ops(ep.graph)
        and not _contains_fake_inputs(eager_args, eager_kwargs)
        and not _contains_fake_weights(model)
    ):
        with PreDispatchSchemaCheckMode():
            model(*eager_args, **eager_kwargs)

    return ep

def make_dynamic_cls(cls):
    suffix = "_pre_dispatch_schema"

    cls_prefix = "PreDispatchSchemaCheck"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        suffix,
        mocked_schema_check_export,
        xfail_prop="_expected_failure_pre_dispatch_schema",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class

make_dynamic_cls(test_export.TestExport)

if __name__ == "__main__":
    run_tests()
