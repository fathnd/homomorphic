import triton
import triton.language as tl

# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
    libdevice = tl.extra.cuda.libdevice
    math = tl.math
elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
    libdevice = tl.extra.intel.libdevice
    math = tl.math
else:
    libdevice = tl.math
    math = tl


if hasattr(tl, "to_tensor"):
    promote_to_tensor = tl.to_tensor
else:

    @triton.jit
    def promote_to_tensor(x):
        # Where promotes without effecting the value in any way
        return tl.where(True, x, x)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def _prod_accumulate(a, b):
    return a * b


@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def minimum(a, b):
    mask = a < b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def maximum(a, b):
    mask = a > b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)


@triton.jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = libdevice.fma(delta, value - new_mean, m2)
    return new_mean, new_m2, new_weight


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        libdevice.fma(delta, w2_over_w, mean_1),
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def welford(mean, m2, weight, dim):
    return tl.reduce((mean, m2, weight), dim, welford_combine)


@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r


@triton.jit
def randint64(seed, offset, low, high):
    r0, r1, r2, r3 = tl.randint4x(seed, offset)
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    result = r0 | (r1 << 32)
    size = high - low
    result = result % size.to(tl.uint64)
    result = result.to(tl.int64) + low
    return result


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def any(a, dim):
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def bucketize_binary_search(
    values,  # 1D tensor
    offsets_ptr,
    indexing_dtype,
    right,  # bool: if true, use intervals closed on the left; see [Note: Inductor bucketize op]
    OFFSETS_SIZE: int,
    BLOCK_SHAPE,  # tuple/list of block shape
):
    """
    See [Note: Inductor bucketize op]
    """

    low = tl.zeros(BLOCK_SHAPE, dtype=indexing_dtype)
    high = tl.full(BLOCK_SHAPE, OFFSETS_SIZE, dtype=indexing_dtype)

    full_range = OFFSETS_SIZE + 1
    while full_range > 1:
        mid = (high + low) // 2
        mask = mid < OFFSETS_SIZE
        bucket_upper_bound = tl.load(offsets_ptr + mid, mask=mask)
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound

        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)

        full_range = (full_range + 1) // 2

    return low


@triton.jit
def pack_value_flag(
    value,
    flag,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE_AS_UINT = tl.core._constexpr_to_value(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    uv = value.to(DTYPE_VALUE_AS_UINT, bitcast=True).to(DTYPE_PACK)
    return flag.to(DTYPE_PACK) | (uv << bitwidth)


@triton.jit
def unpack_value(
    pack,
    DTYPE_VALUE,
    DTYPE_VALUE_AS_UINT,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE = tl.core._constexpr_to_value(DTYPE_VALUE)
    DTYPE_VALUE_AS_UINT = tl.core._constexpr_to_value(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    value_uint = (pack >> bitwidth).to(DTYPE_VALUE_AS_UINT)
    return value_uint.to(DTYPE_VALUE, bitcast=True)


@triton.jit
def unpack_flag(pack, DTYPE_FLAG):
    return pack.to(DTYPE_FLAG)


@triton.jit
def exclusive_scan_decoupled_lookback(
    scratch_base,
    block_value,
    index,
    combine_fn,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    DTYPE_VALUE_AS_UINT: A tl.uint{n} type equal in size to ``block_value``
    DTYPE_PACK: Unsigned type twice the width of block_value

    NOTE: This function is limited to values which are 32-bits or less because
    we need to pack (value, flag) into a single unsigned int.
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    DTYPE_VALUE = block_value.dtype
    pack = pack_value_flag(
        block_value,
        tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        # tl.atomic_load
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack, DTYPE_VALUE_AS_UINT)

        value = unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    pack = pack_value_flag(
        inclusive_prefix,
        tl.full([], 2, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    return exclusive_prefix


@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identiy of combine_fn
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    if index > 0:
        block_value_u64 = block_value.to(tl.uint64, bitcast=True)
        tl.store(scratch_base + 3 * index + 1, block_value_u64)
        tl.debug_barrier()
        flag_one = tl.full([], 1, tl.uint64)
        tl.atomic_xchg(scratch_base + 3 * index + 0, flag_one, sem="release")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], block_value.dtype)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        flag = tl.full([], 0, tl.uint64)
        while flag == 0:
            flag = tl.atomic_add(scratch_base + 3 * test_target + 0, 0, sem="acquire")

        value_u64 = tl.load(scratch_base + 3 * test_target + flag.to(tl.int32))
        value = value_u64.to(block_value.dtype, bitcast=True)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    inclusive_prefix_u64 = inclusive_prefix.to(tl.uint64, bitcast=True)
    tl.store(scratch_base + 3 * index + 2, inclusive_prefix_u64)
    tl.debug_barrier()
    flag_two = tl.full([], 2, tl.uint64)
    tl.atomic_xchg(scratch_base + 3 * index + 0, flag_two, sem="release")

    return exclusive_prefix


@triton.jit
def frexp(x):
    # TODO(isuruf): use inline_asm_elementwise here
    y = libdevice.ilogb(x) + 1
    exponent = tl.where(x == 0, 0, y)
    mantissa = tl.where(x == 0, 0, libdevice.ldexp(x, -y))
    return mantissa, exponent


@triton.jit
def precise_div(a, b):
    # IEEE compliant division
    common_type = (a + b).dtype
    a = promote_to_tensor(a).to(common_type)
    b = promote_to_tensor(b).to(common_type)
    a, b = tl.broadcast(a, b)
    if common_type == tl.float32:
        return tl.math.div_rn(a, b)
    if common_type == tl.float64:
        return libdevice.div_rn(a, b)
    return a / b
