from typing import Optional, Tuple

import torch
from torch._refs import _unsqueeze_multiple
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from torch.library import impl, Library

# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_DTYPE_TO_QVALUE_BOUNDS = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
}

# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"

quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    return torch.empty_like(input, dtype=dtype)

quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    return torch.empty_like(input, dtype=dtype)

# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor2(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype)

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    return quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(out_dtype) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")

@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    return torch.empty_like(input, dtype=out_dtype)

quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype, out_dtype=out_dtype)

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")

# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor2(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(
        input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype, out_dtype=out_dtype)

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
        input,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype)

quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
        input: torch.Tensor,
        qmin: int,
        qmax: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       =================> Documentation says return float
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, \
        f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)

    # scale from determine_qparams returns double
    scale, zp = determine_qparams(
        min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False)
    return scale, zp

quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "CompositeExplicitAutograd")
def choose_qparams_symmetric_tensor(
        input: torch.Tensor,
        qmin: int,
        qmax: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, \
        f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric
    )

@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
        input: torch.Tensor,
        quant_min: int,
        quant_max: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.dtype != torch.float32:
        breakpoint()
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: \
        {quant_min} max: {quant_max}"
    # This takes float32 input but emits double
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)

@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
def choose_qparams_symmetric_tensor_meta(
        input: torch.Tensor,
        quant_min: int,
        quant_max: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    # This takes float32 input but emits double
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)

# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list

quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input)

    for i in range(input.size(0)):
        res[i] = torch.clamp(
            torch.round(input[i] * (1.0 / scales[i])) + zero_points[i],
            quant_min,
            quant_max
        )

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor],
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=out_dtype)

    for i in range(input.size(0)):
        zp = zero_points[i] if zero_points is not None else 0
        # TODO: investigate why
        # (input[i] - zero_points[i]).to(out_dtype) * scales[i]
        # failed the test
        res[i] = (input[i].to(out_dtype) - zp) * scales[i]

    out = res.permute(tuple(permute_axis_list))
    return out

@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor],
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    if out_dtype is None:
        out_dtype = torch.float32
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=out_dtype)

quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor")

class FakeQuantPerChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        with torch._C._AutoDispatchBelowAutograd():
            if input.dtype == torch.bfloat16:
                input = input.to(torch.float32)
            if scales.dtype != torch.float32:
                scales = scales.to(torch.float32)
            if zero_points.dtype != torch.int32:
                zero_points = zero_points.to(torch.int32)
            assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
            assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
            broadcast_dims = list(range(0, axis)) + list(range(axis + 1, input.ndim))
            unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
            unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
            temp = torch.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
            out = (torch.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points) * unsqueeze_scales
            mask = torch.logical_and((temp >= quant_min), (temp <= quant_max))

        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, gy):
        mask, = ctx.saved_tensors
        return gy * mask, None, None, None, None, None

@impl(quantized_decomposed_lib, "fake_quant_per_channel", "AutogradCPU")
def fake_quant_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
) -> torch.Tensor:
    return FakeQuantPerChannel.apply(input, scales, zero_points, axis, quant_min, quant_max)

@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
) -> torch.Tensor:
    return torch.empty_like(input)
