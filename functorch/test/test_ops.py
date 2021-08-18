# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
import torch.nn.functional as F
from torch import Tensor
import functools
import itertools
import copy
import warnings
import unittest
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma
from torch.testing._internal.common_device_type import ops, onlyCPU
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from common_utils import (
    parameterized,
    instantiate_parameterized_methods,
    get_fallback_and_vmap_exhaustive,
    get_exhaustive_batched_inputs,
    opinfo_in_dict,
    xfail,
    skipOps,
)
import types
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from functorch import grad, vjp, vmap
from functorch._src.eager_transforms import _as_tuple

# Version of autograd.grad that handles outputs that don't depend on inputs
def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    inputs, inputs_spec = tree_flatten(inputs)
    result = [torch.zeros_like(inp) for inp in inputs]
    diff_argnums = tuple(i for i, inp in enumerate(inputs) if inp.requires_grad)
    inputs = tuple(inputs[i] for i in diff_argnums)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        something = [(out, go) for out, go in zip(outputs, grad_outputs)
                     if out.requires_grad]
        if len(something) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*something)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph,
                                      allow_unused=True)
    grad_inputs = tuple(torch.zeros_like(inp) if gi is None else gi
                        for gi, inp in zip(grad_inputs, inputs))
    for idx, grad_inp in zip(diff_argnums, grad_inputs):
        result[idx] = grad_inp
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg):
    if is_iterable_of_tensors(arg):
        if all([a.requires_grad for a in arg]):
            return True
        if all([not a.requires_grad for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and arg.requires_grad


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_for_vjp2(f, args, kwargs, output_process_fn_grad=None):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(i for i, arg in enumerate(flat_args) if diff_arg(arg))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


def normalize_op_for_vjp(f, sample):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_for_vjp2(f, args, sample.kwargs, sample.output_process_fn_grad)


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_for_vjp(f, sample)
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result))
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        result, vjp_fn = vjp(fn, *primals)
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        return vjp_fn(cotangents)

    return wrapped, args


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    xfail('linalg.cholesky'),
    xfail('linalg.inv'),
    xfail('linalg.matrix_power'),
    xfail('tensor_split'),
    xfail('to_sparse'),
}

class TestOperators(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_grad', vjp_fail)
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return result.sum()
                result = sum([res.sum() for res in result])
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)

            self.assertEqual(result, expected)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjp', vjp_fail)
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_for_vjp(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            out, vjp_fn = vjp(fn, *primals)
            self.assertEqual(out, result)
            result_vjps = vjp_fn(cotangents)

            _, vjp_fn = ref_vjp(fn, *primals)
            expected_vjps = vjp_fn(cotangents)

            self.assertEqual(result_vjps, expected_vjps)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvjp', vjp_fail)
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            # Compute vjp of vjp
            _, vjp_fn = vjp(fn, *args)
            result_vjps = vjp_fn(cotangents)

            # Compute ref_vjp of vjp. We could have done ref_vjp of ref_vjp,
            # but since we're confident that vjp works by itself, this is
            # an equivalent way to test that.
            _, vjp_fn = ref_vjp(fn, *args)
            expected_vjps = vjp_fn(cotangents)

            self.assertEqual(result_vjps, expected_vjps)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    def test_vmapvjpvjp(self, device, dtype, op):
        self.skipTest("Skipped; these tests take too long")
        op_skip = set({
        })
        op_skip = op_skip.union(vjp_fail)
        if op.name in op_skip:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents, _ = tree_flatten(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result, _ = tree_flatten(result)
                result_vjps, _ = tree_flatten(result_vjps)
                return (*result, *result_vjps)

            for loop_out, batched_out in \
                    get_fallback_and_vmap_exhaustive(vjp_of_vjp, args_and_cotangents, {}):
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapvjp', vjp_fail.union({
        xfail('clamp', ''),
        xfail('diag_embed'),
        xfail('eig'),
        xfail('fft.ihfft'),
        xfail('fft.rfft'),
        xfail('fft.rfftn'),
        xfail('fmax'),
        xfail('fmin'),
        xfail('index_add'),
        xfail('index_copy'),
        xfail('index_fill'),
        xfail('linalg.det', ''),
        xfail('linalg.eigh'),
        xfail('linalg.eigvals', device_type='cuda'),
        xfail('linalg.householder_product'),
        xfail('linalg.matrix_norm'),
        xfail('linalg.norm'),
        xfail('linalg.slogdet'),
        xfail('log_softmax'),
        xfail('logdet'),
        xfail('lu'),
        xfail('lu_unpack'),
        xfail('masked_fill'),
        xfail('masked_scatter'),
        xfail('max', 'reduction_no_dim', device_type='cpu'),
        xfail('median', device_type='cpu'),
        xfail('min', 'reduction_no_dim', device_type='cpu'),
        xfail('nanmedian', device_type='cpu'),
        xfail('nanquantile'),
        xfail('nn.functional.pad', 'circular'),
        xfail('norm', 'fro'),
        xfail('norm', 'nuc'),
        xfail('prod'),
        xfail('put'),
        xfail('quantile'),
        xfail('symeig'),
        xfail('t', device_type='cuda'),
        xfail('take'),
        xfail('unfold'),
        xfail('view_as_complex'),
    }))
    def test_vmapvjp(self, device, dtype, op):
        # These are too annoying to put into the list above
        if op.name in {'nn.functional.linear', 'nn.functional.conv2d'}:
            self.skipTest("Skipped! ExpectedF failures")
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}):
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvmap', vjp_fail.union({
        xfail('__getitem__'),
        xfail('broadcast_to'),
        xfail('clamp', ''),
        xfail('dsplit'),
        xfail('fill_'),
        xfail('gradient'),
        xfail('hsplit'),
        xfail('nn.functional.pad', 'circular'),
        xfail('positive'),
        xfail('ravel'),
        xfail('resolve_conj'),
        xfail('resolve_neg'),
        xfail('unfold'),
        xfail('vsplit'),
    }))
    def test_vjpvmap(self, device, dtype, op):
        if not op.supports_autograd:
            # If the op doesn't support autograd, vmap(op) won't either
            self.skipTest("Skipped! Autograd not supported.")
            return

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            for batched_args, in_dims, kwargs in get_exhaustive_batched_inputs(args, kwargs):
                vmapped_op = vmap(op, in_dims)
                fn, primals = normalize_op_for_vjp2(vmapped_op, batched_args, kwargs,
                                                    sample.output_process_fn_grad)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                _, vjp_fn = vjp(fn, *primals)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
