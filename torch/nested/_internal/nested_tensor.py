from typing import Tuple

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.fx.experimental.sym_node import SymNode, NestedIntNode
from torch.nested._internal import union_find
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_symint_registry = WeakTensorKeyDictionary()


def get_nested_symint(tensor, coeff=1, for_hint=False):
    if isinstance(tensor, torch._subclasses.functional_tensor.FunctionalTensor):
        tensor = torch._from_functional_tensor(tensor.elem)
        return get_nested_symint(tensor, coeff=coeff)
    if (isinstance(tensor, torch._subclasses.fake_tensor.FakeTensor)) and not for_hint:
        return tensor.get_nested_int()
    uf = union_find.get_union_find()
    t_id = uf._tensor_int_map.get_int(tensor)
    return torch.SymInt(NestedIntNode(t_id, tensor, coeff))


# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _lengths: Optional[torch.Tensor]
    # NOTE [ Nested ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a nested int
    # denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use nested ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: Tuple[int, ...]
    _strides: Tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
    _metadata_cache: Dict[str, Any]

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            (0,),
            (0,),
            0,
            torch.contiguous_format,
            values.dtype,
            torch.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
        )
        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs):
        super().__init__()
        # Only support jagged for now.
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        assert values.device == offsets.device

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_nested_symint(ragged_source, coeff=1)
        self._ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        if lengths is not None:
            assert B == lengths.shape[0]

        # subtract 1 to convert to values dim space
        r = self._ragged_idx - 1
        self._size = (B, *values.shape[:r], ragged_size, *values.shape[r + 1 :])
        stride = values.stride()
        self._strides = (ragged_size * stride[r], *stride)

        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # holds properties that are computed lazily
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # collapsed ragged dim must always be dynamic
        torch._dynamo.mark_dynamic(self, self._ragged_idx)
        torch._dynamo.mark_dynamic(self._values, self._ragged_idx - 1)

    def values(self):
        # dispatch to get proper view relationship
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    @property
    def _max_seqlen(self):
        if "max_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["max_seqlen"] = _get_sdpa_extreme_seqlen(
                torch.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["max_seqlen"]

    @property
    def _min_seqlen(self):
        if "min_seqlen" not in self._metadata_cache:
            # compute & cache
            self._metadata_cache["min_seqlen"] = _get_sdpa_extreme_seqlen(
                torch.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
        return self._metadata_cache["min_seqlen"]

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = NestedTensor
        args = (self._values, self._offsets)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            # TODO: Don't guard on this!
            "metadata_cache": self._metadata_cache,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride, original_tensors=None):
        # inner tensors: _values, _offsets, [_lengths]
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 3
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        ragged_idx = meta["ragged_idx"]
        ragged_source = offsets if lengths is None else lengths
        # for the functional to fake wrapping, don't need to do anything (since we rely on unwrapping)
        # for the initial fakification? (different, and we've already saved it on the fake tensor)
        # for the grad_output thing (same, and so we need to do this)
        # for the runtime wrapper (different, and we just create a fresh one because we're in eager)
        # the thing is... this is not enough?
        if original_tensors:
            orig_offsets = original_tensors.get("_offsets", None)
            orig_lengths = original_tensors.get("_lengths", None)
            orig_ragged_source = orig_offsets if orig_lengths is None else orig_lengths
            if type(orig_ragged_source) is type(ragged_source):
                from torch.nested._internal import union_find
                assert orig_ragged_source is not None
                uf = union_find.get_union_find()
                uf.merge(ragged_source, orig_ragged_source)
                # Two FakeTensor now share the same NestedInt?!
                # Previously this was not the case. We also don't need to do
                # the union find thing above?
                ragged_source._nested_int_memo = orig_ragged_source._nested_int_memo
                ragged_source._nested_int_memo_vc = ragged_source._version

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=meta["metadata_cache"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(func)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from .ops import jagged_torch_function

        try:
            return jagged_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


# NB: These fake view autograd.Functions are superseded by real view ops. Don't use them!
# TODO: Remove ViewBufferFromNested, ViewNestedFromBuffer, and buffer_from_jagged once the
# internal BC period has passed.


# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO._values, None, None


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )

    # Check that the NT is representable by the jagged layout.
    # Jagged layout represents (B, *, D_0, D_1, ..., D_N), where the only
    # raggedness allowed is for the single dim immediately adjacent to the batch dim.
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)
    if not at_most_first_ragged:
        raise RuntimeError(
            "Cannot represent given tensor list as a nested tensor with the jagged layout. "
            "Note that the jagged layout only represents shapes of the form "
            "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
        )

    # Set properties appropriately.
    values = torch.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        # TODO: An alternative way to construct offsets is to use F.pad. This avoids creating
        # an extra leaf tensor during the forward, potentially resolving compatibility issues.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),
            ]
        )

    ret_nt = nested_view_from_values_offsets(values, offsets)
    ret_nt._metadata_cache = {
        # compute this now since it's easy
        "max_seqlen": max(t.shape[0] for t in tensors),
        "min_seqlen": min(t.shape[0] for t in tensors),
    }
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # Calculate jagged offsets
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = torch.cat(
        [
            start_list + offset_lengths,
            (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0),
        ]
    )

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    # Check if offsets and lengths make it possibly contiguous and return a regular NT
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False

    actual_max_seqlen = int(torch.max(lengths).item())
    min_seqlen = int(torch.min(lengths).item())

    if is_contiguous:
        ret_nt = nested_view_from_values_offsets(
            values[offsets[0] : offsets[-1]], offsets - offsets[0]
        )
    else:
        ret_nt = nested_view_from_values_offsets_lengths(values, offsets, length_list)

    # populate metadata cache with computed seqlen extremes
    ret_nt._metadata_cache = {
        "max_seqlen": actual_max_seqlen,
        "min_seqlen": min_seqlen,
    }

    return (ret_nt, offsets, None if is_contiguous else length_list)


# NB: A dummy arg is required so that NestedTensor.__torch_dispatch__() is invoked
# for _nested_view_from_values_offsets(). Sizes don't matter much, but they shouldn't be
# 0/1 because the dummy can be fake-ified and we want to avoid specializing.
# This arg is otherwise unused.
_dummy_instance: Optional[torch.Tensor] = None


def _nt_view_dummy() -> torch.Tensor:
    global _dummy_instance
    if _dummy_instance is None:
        _dummy_instance = NestedTensor(
            values=torch.zeros(3, 3, device="meta"),
            offsets=torch.zeros(3, device="meta", dtype=torch.int64),
        ).detach()
    return _dummy_instance


def nested_view_from_values_offsets(values, offsets, ragged_idx=1):
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values, offsets, _nt_view_dummy(), None, ragged_idx
    )


def nested_view_from_values_offsets_lengths(values, offsets, lengths, ragged_idx=1):
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values, offsets, _nt_view_dummy(), lengths, ragged_idx
    )
