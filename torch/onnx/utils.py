"""Functions to export models into the ONNX IR format.

These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import io
import itertools
import os
import re
import textwrap
import typing
import warnings
import zipfile
from typing import (
    Any,
    Callable,
    cast,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
    _constants,
    _exporter_states,
    _patch_torch,
    errors,
    symbolic_caffe2,
    symbolic_helper,
    symbolic_registry,
)
from torch.onnx._globals import GLOBALS

__all__ = [
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "disable_apex_o2_state_dict_hook",
    "setup_onnx_logging",
    "exporter_context",
    "export",
    "warn_on_static_input_change",
    "unpack_quantized_tensor",
    "export_to_pretty_string",
    "unconvertible_ops",
    "get_ns_op_name_from_custom_op",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
]


def is_in_onnx_export() -> bool:
    """Returns whether it is in the middle of ONNX export."""
    return GLOBALS.in_onnx_export


# TODO(justinchuby): Remove dependency to this global variable from constant_fold.cpp
# Skip check due to cannot import IValue from torch._C
_params_dict = {}  # type: ignore[var-annotated]


@contextlib.contextmanager
def select_model_mode_for_export(model, mode: _C_onnx.TrainingMode):
    r"""A context manager to temporarily set the training mode of ``model``
    to ``mode``, resetting it when we exit the with-block.

    Args:
        model: Same type and meaning as ``model`` arg to :func:`export`.
        mode: Same type and meaning as ``training`` arg to :func:`export`.
    """
    if not isinstance(mode, _C_onnx.TrainingMode):
        raise TypeError(
            f"'mode' should be a torch.onnx.TrainingMode enum, but got '{type(mode)}'."
        )
    originally_training: bool = False

    if not isinstance(model, torch.jit.ScriptFunction):
        originally_training = model.training

        # ONNX opset 12 has better support for training amenable models, with updated
        # versions of the dropout and batch_norm operators
        if mode == _C_onnx.TrainingMode.TRAINING or (
            mode == _C_onnx.TrainingMode.PRESERVE and originally_training
        ):
            GLOBALS.export_training = True
            if GLOBALS.export_onnx_opset_version < 12:
                warnings.warn(
                    "You are exporting the model in training mode with onnx opset "
                    f"version {GLOBALS.export_onnx_opset_version}. "
                    "Opset versions lower than opset 12 will not be able to export "
                    "nodes such as Dropout and BatchNorm correctly."
                )
        else:
            GLOBALS.export_training = False

        GLOBALS.training_mode = mode
        if mode == _C_onnx.TrainingMode.TRAINING:
            model.train(True)
        elif mode == _C_onnx.TrainingMode.EVAL:
            model.train(False)
        # else mode == _C_onnx.TrainingMode.PRESERVE, do nothing

    try:
        yield
    finally:
        if not (
            isinstance(model, torch.jit.ScriptFunction)
            or mode == _C_onnx.TrainingMode.PRESERVE
        ):
            model.train(originally_training)


@contextlib.contextmanager
def disable_apex_o2_state_dict_hook(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction]
):
    # Apex O2 hook state_dict to return fp16 weights as fp32.
    # Exporter cannot identify them as same tensors.
    # Since this hook is only used by optimizer, it is safe to
    # remove this hook while exporting.
    if not isinstance(model, torch.jit.ScriptFunction):
        model_hooks = {}  # type: ignore[var-annotated]
        for module in model.modules():
            for key, hook in module._state_dict_hooks.items():
                if type(hook).__name__ == "O2StateDictHook":
                    if module not in model_hooks:
                        model_hooks[module] = {}
                    model_hooks[module][key] = hook
            if module in model_hooks:
                for key in model_hooks[module]:
                    module._state_dict_hooks.pop(key)
        try:
            yield
        finally:
            # Add the hooks back
            for module, m_map in model_hooks.items():
                for key, hook in m_map.items():
                    module._state_dict_hooks[key] = hook
    else:
        try:
            yield
        finally:
            pass


@contextlib.contextmanager
def setup_onnx_logging(verbose):
    is_originally_enabled = torch.onnx.is_onnx_log_enabled()
    if is_originally_enabled or verbose:
        torch.onnx.enable_log()
    try:
        yield
    finally:
        if not is_originally_enabled:
            torch.onnx.disable_log()


@contextlib.contextmanager
def exporter_context(model, mode, verbose):
    with select_model_mode_for_export(
        model, mode
    ) as mode_ctx, disable_apex_o2_state_dict_hook(
        model
    ) as apex_ctx, setup_onnx_logging(
        verbose
    ) as log_ctx:
        yield (mode_ctx, apex_ctx, log_ctx)


def export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    export_params: bool = True,
    verbose: bool = False,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
) -> None:
    r"""Exports a model into ONNX format.

    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Args:
        model (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction):
            the model to be exported.
        args (tuple or torch.Tensor):

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that ``model(*args)`` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::

                args = (x,
                        {'y': input_y,
                         'z': input_z})

            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.

            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::

                    torch.onnx.export(
                        model,
                        (x,
                         # WRONG: will be interpreted as named arguments
                         {y: z}),
                        "test.onnx.pb")

                Write::

                    torch.onnx.export(
                        model,
                        (x,
                         {y: z},
                         {}),
                        "test.onnx.pb")

        f: a file-like object (such that ``f.fileno()`` returns a file descriptor)
            or a string containing a file name.  A binary protocol buffer will be written
            to this file.
        export_params (bool, default True): if True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field ``doc_string``` from the exported model which mentions the source code locations
            for ``model``. If True, ONNX exporter logging will be turned on.
        training (enum, default TrainingMode.EVAL):
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode if model.training is
              False and in training mode if model.training is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations
              which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default OperatorExportTypes.ONNX):

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
              (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
              to standard ONNX ops in the default opset domain. If unable to do so
              (e.g. because support has not been added to convert a particular torch op to ONNX),
              fall back to exporting the op into a custom opset domain without conversion. Applies
              to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
              as well as ATen ops. For the exported model to be usable, the runtime must support
              these non-standard ops.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace "aten")
              are exported as ATen ops (in opset domain "org.pytorch.aten").
              `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch's built-in tensor library, so
              this instructs the runtime to use PyTorch's implementation of these ops.

              .. warning::

                Models exported this way are probably runnable only by Caffe2.

              This may be useful if the numeric differences in implementations of operators are
              causing large differences in behavior between PyTorch and Caffe2 (which is more
              common on untrained models).

            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op
              (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
              (e.g. because support has not been added to convert a particular torch op to ONNX),
              fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
              context.
              For example::

                graph(%0 : Float):
                  %3 : int = prim::Constant[value=0]()
                  # conversion unsupported
                  %4 : Float = aten::triu(%0, %3)
                  # conversion supported
                  %5 : Float = aten::mul(%4, %0)
                  return (%5)

              Assuming ``aten::triu`` is not supported in ONNX, this will be exported as::

                graph(%0 : Float):
                  %1 : Long() = onnx::Constant[value={0}]()
                  # not converted
                  %2 : Float = aten::ATen[operator="triu"](%0, %1)
                  # converted
                  %3 : Float = onnx::Mul(%2, %0)
                  return (%3)

              If PyTorch was built with Caffe2 (i.e. with ``BUILD_CAFFE2=1``), then
              Caffe2-specific behavior will be enabled, including special support
              for ops are produced by the modules described in
              `Quantization <https://pytorch.org/docs/stable/quantization.html>`_.

              .. warning::

                Models exported this way are probably runnable only by Caffe2.

        opset_version (int, default 13): The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. Must be >= 7 and <= 16.
        do_constant_folding (bool, default True): Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        dynamic_axes (dict<string, dict<int, string>> or dict<string, list(int)>, default empty dict):

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
              ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
              list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)

                torch.onnx.export(SumModule(), (torch.ones(2, 2),), "onnx.pb",
                                  input_names=["x"], output_names=["sum"])

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(SumModule(), (torch.ones(2, 2),), "onnx.pb",
                                  input_names=["x"], output_names=["sum"],
                                  dynamic_axes={
                                      # dict value: manually named axes
                                      "x": {0: "my_custom_axis_name"},
                                      # list value: automatic names
                                      "sum": [0],
                                  })

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

            If ``opset_version < 9``, initializers MUST be part of graph
            inputs and this argument will be ignored and the behavior will be
            equivalent to setting this argument to True.

            If None, then the behavior is chosen automatically as follows:

            * If ``operator_export_type=OperatorExportTypes.ONNX``, the behavior is equivalent
              to setting this argument to False.
            * Else, the behavior is equivalent to setting this argument to True.

        custom_opsets (dict<str, int>, default empty dict): A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument.

        export_modules_as_functions (bool or set of type of nn.Module, default False): Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes.

            1. Annotated attributes: class variables that have type annotations via
            `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_
            will be exported as attributes.
            Annotated attributes are not used inside the subgraph of ONNX local function because
            they are not created by PyTorch JIT tracing, but they may be used by consumers
            to determine whether or not to replace the function with a particular fused kernel.

            2. Inferred attributes: variables that are used by operators inside the module. Attribute names
            will have prefix "inferred::". This is to differentiate from predefined attributes retrieved from
            python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.

            * ``False``(default): export ``nn.Module`` forward calls as fine grained nodes.
            * ``True``: export all ``nn.Module`` forward calls as local function nodes.
            * Set of type of nn.Module: export ``nn.Module`` forward calls as local function nodes,
              only if the type of the ``nn.Module`` is found in the set.

    Raises:
      CheckerError: If the ONNX checker detects an invalid ONNX graph. Will still export the
        model to the file ``f`` even if this is raised.
    """

    _export(
        model,
        args,
        f,
        export_params,
        verbose,
        training,
        input_names,
        output_names,
        operator_export_type=operator_export_type,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
        export_modules_as_functions=export_modules_as_functions,
    )


def _is_constant_tensor_list(node):
    if node.kind() != "prim::Constant":
        return False
    output_type = node.output().type()
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True


# ONNX can't handle constants that are lists of tensors, which can
# get generated in constant prop. So we split them back into prim::ListConstructs


def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if _is_constant_tensor_list(node):
            inputs = []
            for val in node.output().toIValue():
                input = g.insertConstant(val)
                input.node().moveBefore(node)
                input.node().copyMetadata(node)
                inputs.append(input)

            lc = (
                g.create("prim::ListConstruct", inputs)
                .insertBefore(node)
                .output()
                .setType(_C.ListType.ofTensors())
            )
            lc.node().copyMetadata(node)
            node.output().replaceAllUsesWith(lc)


def _optimize_graph(
    graph: _C.Graph,
    operator_export_type: _C_onnx.OperatorExportTypes,
    _disable_torch_constant_prop: bool = False,
    fixed_batch_size: bool = False,
    params_dict=None,
    dynamic_axes=None,
    input_names=None,
    module=None,
):
    # Inline everything
    _C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    _C._jit_pass_inline_fork_wait(graph)
    _C._jit_pass_lint(graph)
    _C._jit_pass_onnx_autograd_function_process(graph)
    _C._jit_pass_lower_all_tuples(graph)

    # we now record some ops like ones/zeros
    # into a trace where we previously recorded constants.
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    if _disable_torch_constant_prop is False:
        _C._jit_pass_constant_propagation(graph)

    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    _C._jit_pass_dce(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_canonicalize_graph_fuser_ops(graph)
    _C._jit_pass_lint(graph)
    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_fuse_addmm(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_lower_all_tuples(graph)
    # in _jit_pass_onnx, symbolic functions are called for each node for conversion.
    # However, there are nodes that cannot be converted without additional context.
    # For example, the number of outputs from split (and whether it is static or dynamic) is unknown
    # until the point where it is unpacked by listUnpack node.
    # This pass does a preprocess, and prepares the nodes such that enough context can be received
    # by the symbolic function.
    _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
    _C._jit_pass_onnx_preprocess(graph)

    # onnx does not support tuples, so try to remove them
    _C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    _C._jit_pass_prepare_division_for_onnx(graph)

    _C._jit_pass_onnx_remove_print(graph)
    _C._jit_pass_onnx_preprocess_caffe2(graph)

    symbolic_helper._quantized_ops.clear()
    # Unpack quantized weights for conv and linear ops and insert into graph.
    _C._jit_pass_onnx_unpack_quantized_weights(
        graph, params_dict, symbolic_helper.is_caffe2_aten_fallback()
    )
    if symbolic_helper.is_caffe2_aten_fallback():
        # Insert permutes before and after each conv op to ensure correct order.
        _C._jit_pass_onnx_quantization_insert_permutes(graph, params_dict)

        # Find consecutive permutes that are no-ops and remove them.
        _C._jit_pass_custom_pattern_based_rewrite_graph(
            textwrap.dedent(
                """\
                graph(%Pi):
                    %Pq = quantized::nhwc2nchw(%Pi)
                    %Pr = quantized::nchw2nhwc(%Pq)
                    return (%Pr)"""
            ),
            textwrap.dedent(
                """\
                graph(%Ri):
                    return (%Ri)"""
            ),
            graph,
        )

    # onnx only supports tensors, so we turn all out number types into tensors
    _C._jit_pass_erase_number_types(graph)
    if GLOBALS.onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        _C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    _C._jit_pass_onnx_lint(graph)
    graph = _C._jit_pass_onnx(graph, operator_export_type)
    _C._jit_pass_onnx_lint(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_scalar_type_analysis(
        graph, True, GLOBALS.export_onnx_opset_version
    )
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_peephole(
        graph, GLOBALS.export_onnx_opset_version, fixed_batch_size
    )
    _C._jit_pass_lint(graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    _C._jit_pass_lint(graph)
    graph = _C._jit_pass_canonicalize(graph)
    _C._jit_pass_lint(graph)
    if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
    return graph


def warn_on_static_input_change(input_states):
    """Warns that changes to input dictionaries and strings won't take effect in the traced ONNX graph.

    We accept dictionaries and strings as ONNX inputs, but they should be only for
    configuration use. we detect here if these inputs are modified, and if so we warn
    the user that the changes won't take effect in the traced ONNX graph.
    """
    for input, traced_input in zip(input_states[0], input_states[1]):
        if isinstance(input, dict):
            if list(input.keys()) != list(traced_input.keys()):
                warning = (
                    "We detected that you are modifying a dictionary that is an input to your "
                    "model. "
                    "Note that dictionaries are allowed as inputs in ONNX but they should be "
                    "handled with care. "
                    "Usages of dictionaries is not recommended, and should not be used except "
                    "for configuration use. "
                    "Also note that the order and values of the keys must remain the same. "
                )
                warnings.warn(warning)
        elif isinstance(input, str):
            if input != traced_input:
                warning = (
                    "The model seems to have string inputs/outputs. "
                    "Note that strings will not appear as inputs/outputs of the ONNX graph. "
                )
                warnings.warn(warning)


def _resolve_args_by_export_type(arg_name, arg_value, operator_export_type):
    """Resolves the arguments that are ignored when export_type != operator_export_type.ONNX."""
    if (
        operator_export_type is not operator_export_type.ONNX
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    ):
        if arg_value is True:
            warnings.warn(
                f"'{arg_name}' can be set to True only when 'operator_export_type' is "
                "`ONNX`. Since 'operator_export_type' is not set to 'ONNX', "
                f"'{arg_name}' argument will be ignored."
            )
        arg_value = False
    return arg_value


def _decide_keep_init_as_input(
    keep_initializers_as_inputs: Optional[bool],
    operator_export_type: _C_onnx.OperatorExportTypes,
    opset_version: int,
):
    """Decides whether the initializers in the graph should be listed as ONNX graph inputs.

    This method encapsulates the logic to decide whether the initializers in the graph
    should be listed as ONNX graph inputs (i.e., whether to choose ONNX IR v3 or v4).
    If keep_initializers_as_inputs is not specified (None), then we decide whether to keep
    initializers as graph inputs (val_keep_init_as_ip) based on export type. If export type
    is ONNX, then do not keep initializers as input (val_keep_init_as_ip=False). For all other
    export types keep initializers as input (val_keep_init_as_ip=True).
    If keep_initializers_as_inputs is specified, then respect it. Unless opset version <= 8,
    in which case it must be ignored because for opset version <= 8, all initializers MUST be
    part of graph input (only ONNX IR v3 is allowed), i.e. val_keep_init_as_ip=True.

    Special handling is needed for opset version 8 or lower, because irrespective
    of user input for keep_initializers_as_inputs, the graph must follow ONNX IR v3
    semantics, i.e. all initializers must be listed as ONNX graph input.
    """

    if opset_version < 9:
        if keep_initializers_as_inputs is False:
            warnings.warn(
                "Setting 'keep_initializers_as_inputs=False' for opset version"
                "8 or lower would lead to an invalid ONNX graph. Therefore, "
                "'keep_initializers_as_inputs=False' is ignored during export."
                "Exported model will have initializers as graph inputs (compliant "
                " to ONNX IR v3)."
            )
        return True  # i.e. True == initializers are part of graph input (ONNX IR v3)
    val_keep_init_as_ip = (
        True if keep_initializers_as_inputs is None else keep_initializers_as_inputs
    )
    if (
        keep_initializers_as_inputs is None
        and operator_export_type is _C_onnx.OperatorExportTypes.ONNX
    ):
        val_keep_init_as_ip = False
    return val_keep_init_as_ip


def _decide_add_node_names(add_node_names, operator_export_type):
    return _resolve_args_by_export_type(
        "add_node_names", add_node_names, operator_export_type
    )


def _decide_constant_folding(do_constant_folding, operator_export_type, training):
    do_constant_folding = _resolve_args_by_export_type(
        "do_constant_folding", do_constant_folding, operator_export_type
    )
    if do_constant_folding and (
        training is not None and training is not _C_onnx.TrainingMode.EVAL
    ):
        warnings.warn(
            "It is recommended that constant folding be turned off ('do_constant_folding=False') "
            "when exporting the model in training-amenable mode, i.e. with 'training=TrainingMode.TRAIN' "
            "or 'training=TrainingMode.PRESERVE' (when model is in training mode). Otherwise, some "
            "learnable model parameters may not translate correctly in the exported ONNX model "
            "because constant folding mutates model parameters. Please consider "
            "turning off constant folding or setting the training=TrainingMode.EVAL."
        )
    return do_constant_folding


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _decide_input_format(model, args):
    try:
        sig = _signature(model)
    except ValueError as e:
        warnings.warn(f"{e}, skipping _decide_input_format")
        return args
    try:
        ordered_list_keys = list(sig.parameters.keys())
        if ordered_list_keys[0] == "self":
            ordered_list_keys = ordered_list_keys[1:]
        args_dict: Dict = {}
        if isinstance(args, list):
            args_list = args
        elif isinstance(args, tuple):
            args_list = list(args)
        else:
            args_list = [args]
        if isinstance(args_list[-1], dict):
            args_dict = args_list[-1]
            args_list = args_list[:-1]
        n_nonkeyword = len(args_list)
        for optional_arg in ordered_list_keys[n_nonkeyword:]:
            if optional_arg in args_dict:
                args_list.append(args_dict[optional_arg])
            # Check if this arg has a default value
            else:
                param = sig.parameters[optional_arg]
                if param.default != param.empty:
                    args_list.append(param.default)
        args = args_list if isinstance(args, list) else tuple(args_list)
    # Cases of models with no input args
    except IndexError:
        warnings.warn("No input args, skipping _decide_input_format")
    except Exception as e:
        warnings.warn(f"Skipping _decide_input_format\n {e.args[0]}")

    return args


def _trace(func, args, operator_export_type, return_outs=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args,)

    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        func, args, strict=False, _force_outplace=False, _return_inputs_states=True
    )
    warn_on_static_input_change(inputs_states)

    trace_graph = _optimize_graph(trace_graph, operator_export_type, params_dict={})
    if return_outs:
        return trace_graph, torch_out
    return trace_graph


def _trace_and_get_graph_from_model(model, args):
    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        model, args, strict=False, _force_outplace=False, _return_inputs_states=True
    )
    warn_on_static_input_change(inputs_states)

    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _get_param_count_list(method_graph, args_params):
    param_count_list = []
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        if "PackedParams" in str(input_.type()):
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            param_count_list.append(arg_params_ is not None)

    return param_count_list


def _check_flatten_did_not_remove(original, jit_flattened):
    """torch.jit._flatten removes None. Check if it did so in this case."""

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for inner in x:
                yield from flatten(inner)
        elif isinstance(x, dict):
            for inner in x.values():
                yield from flatten(inner)
        else:
            yield x

    flattened_with_none = list(flatten(original))
    num_none = len(flattened_with_none) - len(jit_flattened)
    assert num_none >= 0
    if num_none:
        raise ValueError(
            f"args contained {num_none} None's after flattening. "
            "When exporting a ScriptModule or ScriptFunction, no args may "
            "be None because that breaks type propagation."
        )


def _create_jit_graph(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction], args: Sequence[Any]
) -> Tuple[
    _C.Graph,
    List[_C.IValue],
    Optional[Any],
    Optional[Union[_C.ScriptModule, _C.ScriptFunction]],
]:
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        _check_flatten_did_not_remove(args, flattened_args)
        torch_out = None

        if isinstance(model, torch.jit.ScriptModule):
            try:
                graph = model.forward.graph
            except AttributeError as e:
                raise RuntimeError("'forward' method must be a script method") from e
            _C._jit_pass_onnx_function_substitution(graph)
            freezed_module = _C._freeze_module(
                cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            method_graph = module._get_method("forward").graph
            args_params = tuple(args) + tuple(params)
            param_count_list = _get_param_count_list(method_graph, args_params)
            in_vars, _ = torch.jit._flatten(args_params)
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            return graph, params, torch_out, module

        # torch.jit.ScriptFunction
        params = []
        graph = model.graph
        _C._jit_pass_onnx_function_substitution(graph)
        param_count_list = _get_param_count_list(graph, args)
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        return graph, params, torch_out, None

    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    _C._jit_pass_onnx_lint(graph)
    state_dict = torch.jit._unique_state_dict(model)
    params = list(state_dict.values())
    graph_inputs = list(graph.inputs())
    user_input_num = len(graph_inputs) - len(state_dict)
    param_names = list(state_dict.keys())
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    _C._jit_pass_onnx_function_substitution(graph)
    return graph, params, torch_out, None


def _get_named_param_dict(graph, params):
    input_and_param_names = [val.debugName() for val in graph.inputs()]
    param_names = input_and_param_names[len(input_and_param_names) - len(params) :]
    _params_dict = dict(zip(param_names, params))
    return _params_dict


def _get_example_outputs(model, args):
    input_args = copy.deepcopy(args)
    input_kwargs = {}
    if input_args and isinstance(input_args[-1], dict):
        input_kwargs = input_args[-1]
        input_args = input_args[:-1]

    example_outputs = model(*input_args, **input_kwargs)
    if isinstance(example_outputs, list):
        example_outputs = [example_outputs]
    elif not isinstance(example_outputs, tuple):
        example_outputs = (example_outputs,)

    return example_outputs


_qtype_vtype_map = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
    torch.quint4x2: torch.int8,
}


def unpack_quantized_tensor(value, cast_onnx_accepted=True):
    if isinstance(value, torch.Tensor) and value.dtype in _qtype_vtype_map:
        q_value_dequantize = value.dequantize()
        q_scale = (
            torch.tensor(value.q_scale(), dtype=torch.double)
            if cast_onnx_accepted
            else torch.tensor(value.q_scale(), dtype=torch.float32)
        )
        q_zero_point = (
            torch.tensor(value.q_zero_point(), dtype=torch.int64)
            if cast_onnx_accepted
            else torch.tensor(value.q_zero_point(), dtype=_qtype_vtype_map[value.dtype])
        )
        q_value = q_value_dequantize / q_scale + q_zero_point
        q_value = q_value.to(dtype=_qtype_vtype_map[value.dtype])
        return q_value, q_scale, q_zero_point
    else:
        return (value,)


def _pre_trace_quant_model(model, args):
    r"""Returns `torch.jit.trace(model, args)` if model is quantized. Otherwise do nothing and return
    original model.

    This is due to https://github.com/pytorch/pytorch/issues/75761.
    """
    if any(
        hasattr(m, "_packed_params") for m in getattr(model, "modules", lambda: [])()
    ) or any(getattr(arg, "is_quantized", False) for arg in args):
        return torch.jit.trace(model, args)
    return model


def _assign_onnx_node_name(graph, node_names):
    """Takes in ONNX graph, and mapping from _C.Node to node name in exported ONNX ModelProto.

    Returns:
        graph (_C.Graph): A TorchScript IR Graph with ONNX nodes, where each _C.Node gets its name
        in exported ONNX ModelProto assigned as attribute ``onnx_name``.
    """

    def n_fn(n, b_fn, node_names):
        for b in n.blocks():
            b_fn(b, node_names)
        if n in node_names:
            n.s_("onnx_name", node_names[n])

    def b_fn(b, node_names):
        for n in b.nodes():
            n_fn(n, b_fn, node_names)

    b_fn(graph, node_names)
    return graph


def _model_to_graph(
    model,
    args,
    verbose=False,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding=True,
    _disable_torch_constant_prop=False,
    fixed_batch_size=False,
    training=_C_onnx.TrainingMode.EVAL,
    dynamic_axes=None,
) -> Tuple[
    _C.Graph,
    Dict[str, torch.Tensor],
    Optional[Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]],
]:
    """Converts model into an ONNX graph.

    Returns:
        graph: A TorchScript IR Graph with ONNX nodes.
        params_dict: Dict from input param name to param value.
        torch_out: The output tensors resulting from the trace of ``model``.
            If ``model`` is a :class:`torch.jit.ScriptModule` or :class:`torch.jit.ScriptFunction`,
            this will be None, since we are not doing any tracing.
    """
    # TODO: can we simplify this to always return a tuple of Tensor or None?

    # Special case for common case of passing a single Tensor
    if isinstance(args, (torch.Tensor, int, float, bool)):
        args = (args,)

    model = _pre_trace_quant_model(model, args)
    graph, params, torch_out, module = _create_jit_graph(model, args)
    params_dict = _get_named_param_dict(graph, params)

    try:
        graph = _optimize_graph(
            graph,
            operator_export_type,
            _disable_torch_constant_prop=_disable_torch_constant_prop,
            fixed_batch_size=fixed_batch_size,
            params_dict=params_dict,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            module=module,
        )
    except Exception as e:
        torch.onnx.log("Torch IR graph at exception: ", graph)
        raise

    is_script = isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule))
    if is_script:
        example_outputs = _get_example_outputs(model, args)
        example_outputs_final = ()
        for example_output in example_outputs:
            example_outputs_final += unpack_quantized_tensor(example_output)
        out_vars, desc = torch.jit._flatten(example_outputs_final)
        _C._jit_pass_onnx_assign_output_shape(
            graph, out_vars, desc, GLOBALS.onnx_shape_inference, is_script
        )

    # NB: ONNX requires complete information about output types, which might be
    # erased by some optimizations, so we need to set it explicitly again.
    else:
        if not isinstance(torch_out, (list, tuple)):
            output_wrapped = [torch_out]
        else:
            output_wrapped = torch_out  # type: ignore[assignment]

        output_tensors, out_desc = _C._jit_flatten(tuple(output_wrapped))
        # assign_output_shape pass is not compatible with quantized outputs.
        # Quantized outputs are flattened to 3 values in ONNX, while packed as
        # single value in PyTorch.
        if not any(getattr(out, "is_quantized", False) for out in output_tensors):
            _C._jit_pass_onnx_assign_output_shape(
                graph,
                output_tensors,
                out_desc,
                GLOBALS.onnx_shape_inference,
                is_script,
            )

    _set_input_and_output_names(graph, input_names, output_names)
    params_dict = _get_named_param_dict(graph, params)

    if training is None or training == _C_onnx.TrainingMode.EVAL:
        params_dict = _C._jit_pass_onnx_eval_peephole(graph, params_dict)

    if (
        do_constant_folding
        and GLOBALS.export_onnx_opset_version in _constants.onnx_constant_folding_opsets
    ):
        params_dict = _C._jit_pass_onnx_constant_fold(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )

    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)

    # For ONNX opset < 9, constants only have three data types: float16, float, double.
    # In this pass transform constants of other data types to float/double + cast operator.
    if GLOBALS.export_onnx_opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    _C._jit_decay_packed_param_input_types(graph)

    # If output names lack a proper name and are identified only by their unique
    # give them a legible name for debugging purposes
    _apply_friendly_debug_names(graph, params_dict)

    return graph, params_dict, torch_out


def export_to_pretty_string(
    model,
    args,
    export_params=True,
    verbose=False,
    training=_C_onnx.TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    export_type=None,
    google_printer=False,
    opset_version=None,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
    add_node_names=True,
    do_constant_folding=True,
    dynamic_axes=None,
):
    r"""
    Similar to :func:`export`, but returns a text representation of the ONNX
    model. Only differences in args listed below. All other args are the same
    as :func:`export`.

    Args:
        add_node_names (bool, default True): Whether or not to set
            NodeProto.name. This makes no difference unless
            ``google_printer=True``.
        google_printer (bool, default False): If False, will return a custom,
            compact representation of the model. If True will return the
            protobuf's `Message::DebugString()`, which is more verbose.

    Returns:
      A UTF-8 str containing a human-readable representation of the ONNX model.
    """
    if opset_version is None:
        opset_version = _constants.onnx_default_opset
    if custom_opsets is None:
        custom_opsets = {}
    symbolic_helper._set_opset_version(opset_version)
    symbolic_helper._set_operator_export_type(operator_export_type)

    with exporter_context(model, training, verbose):
        val_keep_init_as_ip = _decide_keep_init_as_input(
            keep_initializers_as_inputs, operator_export_type, opset_version
        )
        val_add_node_names = _decide_add_node_names(
            add_node_names, operator_export_type
        )
        val_do_constant_folding = _decide_constant_folding(
            do_constant_folding, operator_export_type, training
        )
        args = _decide_input_format(model, args)
        graph, params_dict, torch_out = _model_to_graph(
            model,
            args,
            verbose,
            input_names,
            output_names,
            operator_export_type,
            val_do_constant_folding,
            training=training,
            dynamic_axes=dynamic_axes,
        )

        return graph._pretty_print_onnx(  # type: ignore[attr-defined]
            params_dict,
            opset_version,
            False,
            operator_export_type,
            google_printer,
            val_keep_init_as_ip,
            custom_opsets,
            val_add_node_names,
        )


def unconvertible_ops(
    model, args, training=_C_onnx.TrainingMode.EVAL, opset_version=None
):
    r"""
    Converts the model with operator_export_type set to
    torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH once in order to get a list of
    all the ops that are not supported/implemented by the exporter.

    Args:
        model: Same as corresponding arg to torch.onnx.export.
        args: Same as corresponding arg to torch.onnx.export.
        training: Same as corresponding arg to torch.onnx.export.
        opset_version: Same as corresponding arg to torch.onnx.export.

    Returns:
        Tuple[torch._C.Graph, List[str]], where the list includes the names
        of the unconvertible ops.
    """

    opset_version = opset_version or _constants.onnx_default_opset
    symbolic_helper._set_opset_version(opset_version)
    # operator_export_type is set to ONNX_FALLTHROUGH by default so that if an op is not supported
    # in ONNX, fall through will occur and export the operator as is, as a custom ONNX op.
    with exporter_context(model, training, False):
        args = _decide_input_format(model, args)
        graph, params_dict, torch_out = _model_to_graph(
            model,
            args,
            # So that if an op connot be converted to ONNX, it will be kept
            # as-is rather than cause a failure.
            operator_export_type=_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
    unsupported_ops = list()
    supported_namespaces = ("onnx", "prim", "quantized")
    for node in graph.nodes():
        if node.kind().split(":")[0] not in supported_namespaces:
            unsupported_ops.append(node.kind())
    return graph, unsupported_ops


def _setup_trace_module_map(model, export_modules_as_functions):
    def __setup_trace_module_map():
        trace_module_map = {_m: torch.typename(type(_m)) for _m in model.modules()}
        torch.jit._trace._trace_module_map = trace_module_map
        return trace_module_map

    def __register_attribute_hook():
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return

            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)

            _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        trace_module_map = __setup_trace_module_map()
        export_modules_as_functions = {v for k, v in trace_module_map.items()}
    elif (
        isinstance(export_modules_as_functions, set)
        and len(export_modules_as_functions) > 0
    ):

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be "
                    "passed in the set for argument `export_modules_as_functions`. "
                    "Got `%s`." % (type(v).__name__)
                )

        trace_module_map = __setup_trace_module_map()
        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
        export_modules_as_functions = module_typenames
    else:
        export_modules_as_functions = None

    if export_modules_as_functions:
        __register_attribute_hook()

    return export_modules_as_functions


def _reset_trace_module_map():
    torch.jit._trace._trace_module_map = None
    _C._jit_pass_onnx_clear_scope_records()


def _get_module_attributes(module):

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]
    return {k: getattr(module, k) for k in annotations}


def _export(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=_C_onnx.TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    export_type=None,
    opset_version=None,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    fixed_batch_size=False,
    custom_opsets=None,
    add_node_names=True,
    onnx_shape_inference=True,
    export_modules_as_functions=False,
):
    if export_type is None:
        export_type = _exporter_states.ExportTypes.PROTOBUF_FILE

    if isinstance(model, torch.nn.DataParallel):
        raise ValueError(
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try "
            "torch.onnx.export(model.module, ...)"
        )
    assert GLOBALS.in_onnx_export is False
    GLOBALS.in_onnx_export = True
    try:

        symbolic_helper._set_onnx_shape_inference(onnx_shape_inference)

        if opset_version is None:
            opset_version = _constants.onnx_default_opset

        if export_modules_as_functions and opset_version < 15:
            raise ValueError(
                "`export_modules_as_functions` is not supported for `opset_version` < 15."
                "This is because `opset_version` < 15 implies IR version < 8, which means "
                "no local function support. "
            )
        export_modules_as_functions = _setup_trace_module_map(
            model, export_modules_as_functions
        )

        if not operator_export_type:
            if _C_onnx._CAFFE2_ATEN_FALLBACK:
                operator_export_type = _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            else:
                operator_export_type = _C_onnx.OperatorExportTypes.ONNX

        # By default, training=TrainingMode.EVAL,
        # which is good because running a model in training mode could result in
        # internal buffers getting updated, dropout getting applied, etc.
        # If you really know what you're doing, you can turn
        # training=TrainingMode.TRAINING or training=TrainingMode.PRESERVE,
        # (to preserve whatever the original training mode was.)
        symbolic_helper._set_opset_version(opset_version)
        symbolic_helper._set_operator_export_type(operator_export_type)
        with exporter_context(model, training, verbose):
            val_keep_init_as_ip = _decide_keep_init_as_input(
                keep_initializers_as_inputs, operator_export_type, opset_version
            )
            val_add_node_names = _decide_add_node_names(
                add_node_names, operator_export_type
            )
            val_do_constant_folding = _decide_constant_folding(
                do_constant_folding, operator_export_type, training
            )
            # Normally f can be a file-like object, but for large models, the external data format requires a
            # valid `model_file_location`. Code in export.cpp will enforce this.
            if isinstance(f, str):
                model_file_location = f
            else:
                model_file_location = ""
            args = _decide_input_format(model, args)
            if dynamic_axes is None:
                dynamic_axes = {}
            _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

            graph, params_dict, torch_out = _model_to_graph(
                model,
                args,
                verbose,
                input_names,
                output_names,
                operator_export_type,
                val_do_constant_folding,
                fixed_batch_size=fixed_batch_size,
                training=training,
                dynamic_axes=dynamic_axes,
            )

            # TODO: Don't allocate a in-memory string for the protobuf
            defer_weight_export = (
                export_type is not _exporter_states.ExportTypes.PROTOBUF_FILE
            )
            if custom_opsets is None:
                custom_opsets = {}

            _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
            node_attr_to_name = {}  # type: ignore[var-annotated]
            if export_modules_as_functions:
                # NOTE: cannot call DCE after this pass. DCE will remove function definition nodes.
                node_attr_to_name = _C._jit_pass_onnx_function_extraction(
                    graph, export_modules_as_functions, list(params_dict.keys())
                )
            params_dict = _C._jit_pass_onnx_deduplicate_initializers(  # type: ignore[assignment]
                graph, params_dict, getattr(model, "training", False)  # type: ignore[arg-type]
            )
            if export_params:
                (
                    proto,
                    export_map,
                    val_use_external_data_format,
                    node_names,
                ) = graph._export_onnx(  # type: ignore[attr-defined]
                    params_dict,
                    opset_version,
                    dynamic_axes,
                    defer_weight_export,
                    operator_export_type,
                    not verbose,
                    val_keep_init_as_ip,
                    custom_opsets,
                    val_add_node_names,
                    model_file_location,
                    node_attr_to_name,
                )
            else:
                (
                    proto,
                    export_map,
                    val_use_external_data_format,
                    node_names,
                ) = graph._export_onnx(  # type: ignore[attr-defined]
                    {},
                    opset_version,
                    dynamic_axes,
                    False,
                    operator_export_type,
                    not verbose,
                    val_keep_init_as_ip,
                    custom_opsets,
                    val_add_node_names,
                    model_file_location,
                    node_attr_to_name,
                )
            if verbose:
                torch.onnx.log(
                    "Exported graph: ", _assign_onnx_node_name(graph, node_names)
                )
            if export_type == _exporter_states.ExportTypes.PROTOBUF_FILE:
                assert len(export_map) == 0
                with torch.serialization._open_file_like(f, "wb") as opened_file:
                    opened_file.write(proto)
            elif export_type in [
                _exporter_states.ExportTypes.ZIP_ARCHIVE,
                _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
            ]:
                compression = (
                    zipfile.ZIP_DEFLATED
                    if export_type
                    == _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE
                    else zipfile.ZIP_STORED
                )
                with zipfile.ZipFile(f, "w", compression=compression) as z:
                    z.writestr(_constants.ONNX_ARCHIVE_MODEL_PROTO_NAME, proto)
                    for k, v in export_map.items():
                        z.writestr(k, v)
            elif export_type == _exporter_states.ExportTypes.DIRECTORY:
                if os.path.exists(f):
                    assert os.path.isdir(f)
                else:
                    os.makedirs(f)

                model_proto_file = os.path.join(
                    f, _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME
                )
                with torch.serialization._open_file_like(
                    model_proto_file, "wb"
                ) as opened_file:
                    opened_file.write(proto)

                for k, v in export_map.items():
                    weight_proto_file = os.path.join(f, k)
                    with torch.serialization._open_file_like(
                        weight_proto_file, "wb"
                    ) as opened_file:
                        opened_file.write(v)
            else:
                raise RuntimeError("Unknown export type")

            # The ONNX checker only works for ONNX graph. So if the operator_export_type is not ONNX,
            # we can skip this check.
            # If large model format export is enabled, proto will only contain data location instead of
            # raw data and _check_onnx_proto() will fail because it can only handle the raw ONNX proto
            # string in memory.
            if (operator_export_type is _C_onnx.OperatorExportTypes.ONNX) and (
                not val_use_external_data_format
            ):
                try:
                    _C._check_onnx_proto(proto, full_check=True)
                except RuntimeError as e:
                    raise errors.CheckerError(e)
    finally:
        assert GLOBALS.in_onnx_export
        GLOBALS.in_onnx_export = False
        _reset_trace_module_map()

    return torch_out


def _apply_friendly_debug_names(graph, params):
    for n in graph.nodes():
        for v in n.inputs():
            old_name = v.debugName()
            if old_name != str(v.unique()):
                continue
            new_name = f"{n.kind()}_{v.unique()}"
            v.setDebugName(new_name)
            if old_name in params:
                params[new_name] = params.pop(old_name)


def _set_input_and_output_names(graph, input_names, output_names):
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) exceeded number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list))
            )

        # Mark if the output node DebugName is set before.
        output_node_set = set()
        for i, (name, node) in enumerate(zip(name_list, node_list)):
            # Duplicated output node, insert onnx::Identity to avoid setting the same DebugName after setDebugName().
            if descriptor == "output":
                if node in output_node_set:
                    identity_node = graph.create("onnx::Identity")
                    identity_node.insertAfter(node.node())
                    identity_node.addInput(node)
                    identity_node.output().setType(node.type())
                    graph.return_node().replaceInput(i, identity_node.output())
                    node = identity_node.output()
                output_node_set.add(node)

            if node.debugName() != name:
                node.setDebugName(name)

    set_names(list(graph.inputs()), input_names, "input")
    set_names(list(graph.outputs()), output_names, "output")


def _run_symbolic_method(g, op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(g, *args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = (f"{e.args[0]} (occurred when translating {op_name})",)
        raise


def _add_block(node: _C.Node):
    return node.addBlock()  # type: ignore[attr-defined]


def _add_input_to_block(block: _C.Block):
    return block.addInputToBlock()  # type: ignore[attr-defined]


def _add_output_to_block(block: _C.Block, value: _C.Value):
    new_output = block.registerOutput(value)  # type: ignore[attr-defined]
    return new_output


# Note [Export inplace]
# ~~~~~~~~~~~~~~~~~~~~~
# In abstract, it would be better for us to export inplace annotations,
# than to not export them, since it is useful information that can
# help the target of an ONNX export export more efficiently.  However,
# ONNX doesn't currently formalize inplace. Fortunately, it's sound to drop
# inplace annotations, but we are losing information this way.


def _find_symbolic_in_registry(
    domain: str,
    op_name: str,
    opset_version: int,
    operator_export_type: _C_onnx.OperatorExportTypes,
) -> Optional[Callable]:
    """Looks up for the symbolic function in the registry.

    Args:
        domain: The domain of the symbolic function.
        op_name: The name of the op.
        opset_version: Currect opset used.
        operator_export_type: An enum in _C_onnx.OperatorExportTypes.

    Returns:
        The symbolic function if found, None otherwise.
    """

    if not symbolic_registry.is_registered_op(op_name, domain, opset_version):
        if operator_export_type == _C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH:
            # Use the original node directly
            return None
    return symbolic_registry.get_registered_op(op_name, domain, opset_version)


def _should_aten_fallback(ns, op_name, opset_version, operator_export_type):

    is_exportable_aten_op = symbolic_registry.is_registered_op(
        op_name, "", opset_version
    )
    is_onnx_aten_export = operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN
    is_aten_fallback_export = (
        operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
    return is_onnx_aten_export or (
        not is_exportable_aten_op and is_aten_fallback_export
    )


def _need_symbolic_context(symbolic_fn) -> bool:
    """Checks if the first argument to symbolic_fn is annotated as type `torch.onnx.SymbolicContext`."""
    params = tuple(inspect.signature(symbolic_fn).parameters.values())
    # When the annotation is postpone-evaluated, the annotation is a string
    # and not a type. We need to use get_type_hints to get the real type.
    if not params:
        return False
    first_param_name = params[0].name
    type_hints = typing.get_type_hints(symbolic_fn)
    if first_param_name not in type_hints:
        return False
    param_type = type_hints[first_param_name]
    return issubclass(param_type, _exporter_states.SymbolicContext)


def _get_aten_op_overload_name(n: _C.Node) -> str:

    # Returns `overload_name` attribute to ATen ops on non-Caffe2 builds
    schema = n.schema()
    if not schema.startswith("aten::") or symbolic_helper.is_caffe2_aten_fallback():
        return ""
    return _C.parse_schema(schema).overload_name


def _run_symbolic_function(
    g: _C.Graph,
    block: _C.Block,
    n: _C.Node,
    inputs: Any,
    env: Dict[_C.Value, _C.Value],
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
) -> Optional[Union[_C.Value, Tuple[_C.Value, ...]]]:
    """Runs a symbolic function.

    The function is used in C++ to export the node to ONNX.

    Returns:
        A single or a tuple of Values.
        None when the node gets cloned as is into the new graph.
    """

    opset_version = GLOBALS.export_onnx_opset_version

    # See Note [Export inplace]
    node_kind = n.kind()
    if node_kind.endswith("_"):
        # Treat relu_ -> relu; add_ -> add etc.
        ns_op_name = node_kind[:-1]
    else:
        ns_op_name = node_kind

    namespace, op_name = ns_op_name.split("::")

    try:
        symbolic_registry.register_version("", opset_version)

        # Caffe2-specific: Quantized op symbolics are registered for opset 9 only.
        if symbolic_helper.is_caffe2_aten_fallback() and opset_version == 9:
            symbolic_caffe2.register_quantized_ops("caffe2", opset_version)

        if namespace == "aten":
            domain = ""
        elif namespace == "quantized" and symbolic_helper.is_caffe2_aten_fallback():
            domain = "caffe2"
        else:
            domain = namespace

        if symbolic_registry.is_registered_op(op_name, domain, opset_version):
            symbolic_fn = _find_symbolic_in_registry(
                domain, op_name, opset_version, operator_export_type
            )
            assert symbolic_fn is not None

            attrs = {k: symbolic_helper._node_get(n, k) for k in n.attributeNames()}
            if _need_symbolic_context(symbolic_fn):
                ctx = _exporter_states.SymbolicContext(_params_dict, env, n, block)
                return symbolic_fn(ctx, g, *inputs, **attrs)
            # PythonOp symbolic need access to the node to resolve the name conflict,
            # this is inconsistent with regular op symbolic.
            if op_name == "PythonOp":
                inputs = (n, *inputs)
            return symbolic_fn(g, *inputs, **attrs)
        elif namespace == "onnx":
            # Clone node to trigger ONNX shape inference
            attrs = {
                k + "_" + n.kindOf(k)[0]: symbolic_helper._node_get(n, k)
                for k in n.attributeNames()
            }
            return g.op(op_name, *inputs, **attrs, outputs=n.outputsSize())  # type: ignore[attr-defined]
        elif _should_aten_fallback(
            namespace, op_name, opset_version, operator_export_type
        ):
            # Direct ATen export requested
            attrs = {
                k + "_" + n.kindOf(k)[0]: symbolic_helper._node_get(n, k)
                for k in n.attributeNames()
            }
            outputs = n.outputsSize()
            attrs["outputs"] = outputs
            # `overload_name` is set for non-Caffe2 builds only
            return g.at(  # type: ignore[attr-defined]
                op_name, *inputs, overload_name=_get_aten_op_overload_name(n), **attrs
            )
        else:
            raise errors.UnsupportedOperatorError(
                domain,
                op_name,
                opset_version,
                symbolic_registry.get_op_supported_version(
                    op_name, domain, opset_version
                ),
            )
    except RuntimeError:
        if operator_export_type == _C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH:
            return None
        elif (
            operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            and not symbolic_helper.is_caffe2_aten_fallback()
        ):
            # Emit ATen op for non-Caffe2 builds when `operator_export_type==ONNX_ATEN_FALLBACK`
            attrs = {
                k + "_" + n.kindOf(k)[0]: symbolic_helper._node_get(n, k)
                for k in n.attributeNames()
            }
            return g.at(  # type: ignore[attr-defined]
                op_name, *inputs, overload_name=_get_aten_op_overload_name(n), **attrs
            )
        raise
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = (f"{e.args[0]} \n(Occurred when translating {op_name}).",)
        raise


def get_ns_op_name_from_custom_op(symbolic_name):
    if not bool(
        re.match(r"^[a-zA-Z0-9-_]*::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name)
    ):
        raise ValueError(
            f"Failed to register operator {symbolic_name}."
            "The symbolic name must match the format Domain::Name, "
            "and should start with a letter and contain only "
            "alphanumerical characters"
        )

    ns, op_name = symbolic_name.split("::")
    if ns == "onnx":
        raise ValueError(
            f"Failed to register operator {symbolic_name}. {ns} domain cannot be modified."
        )

    if ns == "aten":
        ns = ""

    return ns, op_name


def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    """Registers a symbolic function for a custom operator.

    When the user registers symbolic for custom/contrib ops,
    it is highly recommended to add shape inference for that operator via setType API,
    otherwise the exported graph may have incorrect shape inference in some extreme cases.
    An example of setType is `test_aten_embedding_2` in `test_operators.py`.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        symbolic_fn (Callable): A function that takes in the ONNX graph and
            the input arguments to the current operator, and returns new
            operator nodes to add to the graph.
        opset_version (int): The ONNX opset version in which to register.
    """
    ns, op_name = get_ns_op_name_from_custom_op(symbolic_name)

    for version in itertools.chain(
        _constants.onnx_stable_opsets, [_constants.onnx_main_opset]
    ):
        if version >= opset_version:
            symbolic_registry.register_op(op_name, symbolic_fn, ns, version)


def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int):
    """Unregisters ``symbolic_name``.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        opset_version (int): The ONNX opset version in which to unregister.
    """
    ns, op_name = get_ns_op_name_from_custom_op(symbolic_name)

    for version in itertools.chain(
        _constants.onnx_stable_opsets, [_constants.onnx_main_opset]
    ):
        if version >= opset_version:
            symbolic_registry.unregister_op(op_name, ns, version)


def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names):
    """Ensures dynamic axes argument is follows the expected format."""
    if len(dynamic_axes) == 0:
        return

    if hasattr(model, "graph"):
        # Extracting set of valid input/output names that shall be used for dynamic_axes
        if (input_names is None) or len(input_names) == 0:
            input_names = [x.debugName() for x in model.graph.inputs()]
        if (output_names is None) or len(output_names) == 0:
            output_names = [y.debugName() for y in model.graph.outputs()]

    valid_names = set((input_names or []) + (output_names or []))

    # If dynamic axes are provided as a list rather than dictionary, they should
    # first get converted to a dictionary in expected format. If desired axes names
    # are not provided for dynamic axes, automatic names shall be generated for
    # provided dynamic axes of specified input/output
    for key, value in dynamic_axes.items():
        if key not in valid_names:
            warnings.warn(
                f"Provided key {key} for dynamic axes is not a valid input/output name"
            )
        if isinstance(value, list):
            warnings.warn(
                "No names were found for specified dynamic axes of provided input."
                f"Automatically generated names will be applied to each dynamic axes of input {key}"
            )

            value_dict = {}
            for i, x in enumerate(value):
                if not isinstance(x, int):
                    raise ValueError(
                        "The type of axis index is expected to be an integer"
                    )
                if x in value_dict:
                    warnings.warn(
                        f"Duplicate dynamic axis index {x} was provided for input {key}."
                    )
                else:
                    value_dict[x] = str(key) + "_dynamic_axes_" + str(i + 1)
            dynamic_axes[key] = value_dict
