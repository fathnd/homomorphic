# Owner(s): ["oncall: export"]

try:
    from . import test_export, test_export_predispatch_schema, testing
except ImportError:
    import test_export
    import test_export_predispatch_schema
    import testing
from torch.export._trace import _export

import copy
import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.schema_check_mode import SchemaCheckMode

test_classes = {}


class TestOpInfo(TestCase):
    # @ops(op_db)
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_schema_check_op(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        inputs = next(sample_inputs_itr)
        args = [inputs.input] + list(inputs.args)
        kwargs = inputs.kwargs
        with test_export_predispatch_schema.PreDispatchSchemaCheckMode():
            op.op(*args, **kwargs)

instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
