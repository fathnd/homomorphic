from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union

import torch

from .fake_quantize import (
    default_weight_fake_quant,
    FixedQParamsFakeQuantize,
)
from .observer import (
    _PartialWrapper,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    default_placeholder_observer,
    default_weight_observer,
)
from .qconfig import (
    default_reuse_input_qconfig,
    default_symmetric_qnnpack_qconfig,
    get_default_qconfig,
    get_default_qat_qconfig,
    QConfig,
    QConfigAny
)


__all__ = [
    "get_default_qconfig_mapping",
    "get_default_qat_qconfig_mapping",
    "QConfigMapping",
]


# TODO: replace all usages with these constants
GLOBAL_DICT_KEY = ""
OBJECT_TYPE_DICT_KEY = "object_type"
MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
MODULE_NAME_DICT_KEY = "module_name"
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"

_FIXED_QPARAMS_OP_TO_OBSERVER: Dict[Union[Callable, str], _PartialWrapper] = {
    torch.nn.Hardsigmoid: default_fixed_qparams_range_0to1_observer,
    torch.nn.functional.hardsigmoid: default_fixed_qparams_range_0to1_observer,
    "hardsigmoid": default_fixed_qparams_range_0to1_observer,
    "hardsigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Sigmoid: default_fixed_qparams_range_0to1_observer,
    torch.sigmoid: default_fixed_qparams_range_0to1_observer,
    "sigmoid": default_fixed_qparams_range_0to1_observer,
    "sigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Softmax: default_fixed_qparams_range_0to1_observer,
    torch.nn.Tanh: default_fixed_qparams_range_neg1to1_observer,
    torch.tanh: default_fixed_qparams_range_neg1to1_observer,
    "tanh": default_fixed_qparams_range_neg1to1_observer,
    "tanh_": default_fixed_qparams_range_neg1to1_observer,
}


def _get_default_qconfig_mapping(is_qat: bool, backend: str, version: int) -> QConfigMapping:
    """
    Return the default QConfigMapping for the given quantization type and backend.
    """
    if is_qat:
        qconfig = get_default_qat_qconfig(backend, version)
    else:
        qconfig = get_default_qconfig(backend, version)
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer

    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer.
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend in ("fbgemm", "x86"):
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight)
    else:
        qconfig_transpose = qconfig

    # currently layernorm only supports float weights
    # we have to add this because otherwise there will be a extra quantize-dequantize pair
    qconfig_layernorm = QConfig(activation=qconfig.activation, weight=default_placeholder_observer)

    qconfig_mapping = QConfigMapping() \
        .set_global(qconfig) \
        .set_object_type("reshape", default_reuse_input_qconfig) \
        .set_object_type(torch.nn.Conv1d, qconfig) \
        .set_object_type(torch.nn.Conv2d, qconfig) \
        .set_object_type(torch.nn.Conv3d, qconfig) \
        .set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose) \
        .set_object_type(torch.nn.Linear, qconfig) \
        .set_object_type(torch.nn.functional.conv1d, qconfig) \
        .set_object_type(torch.nn.functional.conv2d, qconfig) \
        .set_object_type(torch.nn.functional.conv3d, qconfig) \
        .set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.linear, qconfig) \
        .set_object_type(torch.nn.ReLU, qconfig) \
        .set_object_type(torch.nn.functional.relu, qconfig) \
        .set_object_type(torch.relu, qconfig) \
        .set_object_type(torch.nn.BatchNorm1d, qconfig) \
        .set_object_type(torch.nn.BatchNorm2d, qconfig) \
        .set_object_type(torch.nn.BatchNorm3d, qconfig) \
        .set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm) \
        .set_object_type(torch.nn.LayerNorm, qconfig_layernorm) \

    # Use special observers for ops with fixed qparams
    fixed_qparams_observer_to_qconfig: Dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(activation=activation, weight=default_weight)
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)

    return qconfig_mapping

def get_default_qconfig_mapping(backend="fbgemm", version=0) -> QConfigMapping:
    """
    Return the default QConfigMapping for post training quantization.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86", "fbgemm" (default), "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    # TODO: add assert for backend choices
    return _get_default_qconfig_mapping(False, backend, version)

def get_default_qat_qconfig_mapping(backend="fbgemm", version=1) -> QConfigMapping:
    """
    Return the default QConfigMapping for quantization aware training.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86", "fbgemm" (default), "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    return _get_default_qconfig_mapping(True, backend, version)

def _get_symmetric_qnnpack_qconfig_mapping():
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qconfig`
    as the default QConfig.
    """
    qconfig_mapping = get_default_qconfig_mapping("qnnpack") \
        .set_global(default_symmetric_qnnpack_qconfig)
    for pattern in qconfig_mapping.object_type_qconfigs.keys():
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, default_symmetric_qnnpack_qconfig)
    return qconfig_mapping

class QConfigMapping:
    """
    Mapping from model ops to :class:`torch.ao.quantization.QConfig` s.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfig

        ``set_object_type`` : sets the QConfig for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfig for modules matching the given regex string

        ``set_module_name`` : sets the QConfig for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfig for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Example usage::

        qconfig_mapping = QConfigMapping()
            .set_global(global_qconfig)
            .set_object_type(torch.nn.Linear, qconfig1)
            .set_object_type(torch.nn.ReLU, qconfig1)
            .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
            .set_module_name_regex("foo.*", qconfig2)
            .set_module_name("module1", qconfig1)
            .set_module_name("module2", qconfig2)
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)

    """

    def __init__(self):
        # In increasing match priority:
        self.global_qconfig: QConfigAny = None
        self.object_type_qconfigs: OrderedDict[Union[Callable, str], QConfigAny] = OrderedDict()
        self.module_name_regex_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()
        self.module_name_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()
        self.module_name_object_type_order_qconfigs: OrderedDict[Tuple[str, Callable, int], QConfigAny] =\
            OrderedDict()

    def set_global(self, global_qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the global (default) QConfig.
        """
        self.global_qconfig = global_qconfig
        return self

    def set_object_type(self, object_type: Union[Callable, str], qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for a given module type, function, or method name.
        If the QConfig for an existing object type was already set, the new QConfig will override the old one.
        """
        self.object_type_qconfigs[object_type] = qconfig
        return self

    def set_module_name_regex(self, module_name_regex: str, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching the given regex string.

        Regexes will be matched in the order in which they are registered through this method.
        Thus, the caller should register more specific patterns first, e.g.::

            qconfig_mapping = QConfigMapping()
                .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
                .set_module_name_regex("foo.*bar.*", qconfig2)
                .set_module_name_regex("foo.*", qconfig3)

        In this example, "foo.bar.conv0" would match qconfig1, "foo.bar.linear" would match qconfig2,
        and "foo.baz.relu" would match qconfig3.

        If the QConfig for an existing module name regex was already set, the new QConfig will override the
        old one while preserving the order in which the regexes were originally registered.
        """
        self.module_name_regex_qconfigs[module_name_regex] = qconfig
        return self

    def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching the given module name.
        If the QConfig for an existing module name was already set, the new QConfig will override the old one.
        """
        self.module_name_qconfigs[module_name] = qconfig
        return self

    def set_module_name_object_type_order(
            self,
            module_name: str,
            object_type: Callable,
            index: int,
            qconfig: QConfigAny) -> QConfigMapping:
        """
        Set the QConfig for modules matching a combination of the given module name, object type,
        and the index at which the module appears.

        If the QConfig for an existing (module name, object type, index)  was already set, the new QConfig
        will override the old one.
        """
        self.module_name_object_type_order_qconfigs[(module_name, object_type, index)] = qconfig
        return self

    # TODO: remove this
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``QConfigMapping`` to a dictionary with the following keys:

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are lists of tuples.
        """
        return {
            GLOBAL_DICT_KEY: self.global_qconfig,
            OBJECT_TYPE_DICT_KEY: list(self.object_type_qconfigs.items()),
            MODULE_NAME_REGEX_DICT_KEY: list(self.module_name_regex_qconfigs.items()),
            MODULE_NAME_DICT_KEY: list(self.module_name_qconfigs.items()),
            MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: [
                (*k, v) for k, v in self.module_name_object_type_order_qconfigs.items()
            ],
        }

    # TODO: remove this
    @classmethod
    def from_dict(cls, qconfig_dict: Dict[str, Any]) -> QConfigMapping:
        """
        Create a ``QConfigMapping`` from a dictionary with the following keys (all optional):

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        """
        conf = cls()
        if GLOBAL_DICT_KEY in qconfig_dict:
            conf.set_global(qconfig_dict[GLOBAL_DICT_KEY])
        for object_type, qconfig in qconfig_dict.get(OBJECT_TYPE_DICT_KEY, []):
            conf.set_object_type(object_type, qconfig)
        for module_name_regex, qconfig in qconfig_dict.get(MODULE_NAME_REGEX_DICT_KEY, []):
            conf.set_module_name_regex(module_name_regex, qconfig)
        for module_name, qconfig in qconfig_dict.get(MODULE_NAME_DICT_KEY, []):
            conf.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_dict.get(MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []):
            conf.set_module_name_object_type_order(module_name, object_type, index, qconfig)
        return conf
