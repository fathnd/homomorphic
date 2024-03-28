import contextlib
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from ._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
)
from ._fsdp_param import FSDPParam


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    all_gather_input_numels: List[int]


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.cuda.Stream,
    all_gather_stream: torch.cuda.Stream,
    device: torch.device,
) -> Optional[AllGatherResult]:
    world_size, rank = group.size(), group.rank()
    # - Copy in
    ctx = contextlib.nullcontext()
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        ctx = torch.cuda.stream(all_gather_copy_in_stream)
    with ctx:
        param_all_gather_inputs = [
            fsdp_param.all_gather_input for fsdp_param in fsdp_params
        ]
        dtype = param_all_gather_inputs[0].dtype
        if not all(t.dtype == dtype for t in param_all_gather_inputs):
            raise NotImplementedError(
                f"Mixed dtype not supported yet: {[t.dtype for t in param_all_gather_inputs]}"
            )
        inp_split_sizes = [inp.numel() for inp in param_all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_output = torch.empty(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device
        )
        all_gather_input = all_gather_output.narrow(
            0, all_gather_input_numel * rank, all_gather_input_numel
        )
        foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
        with torch.no_grad():
            torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
        del param_all_gather_inputs
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        all_gather_stream.wait_stream(all_gather_copy_in_stream)
    ctx = contextlib.nullcontext()
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        ctx = torch.cuda.stream(all_gather_stream)
    with ctx:
        # - All-gather
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            all_gather_event = all_gather_stream.record_event()
        else:
            all_gather_event = None
        return AllGatherResult(
            all_gather_output, all_gather_event, all_gather_work, inp_split_sizes
        )


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        all_gather_input_numels,
    ) = all_gather_result
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        if all_gather_event is not None:  # sync op
            torch.cuda.current_stream().wait_event(all_gather_event)
        if all_gather_work is not None:  # async op
            all_gather_work.wait()
    world_size = group.size()
    dtype, device = all_gather_output.dtype, all_gather_output.device
    for all_gather_input_numel, fsdp_param in zip(all_gather_input_numels, fsdp_params):
        fsdp_param.init_all_gather_output(
            all_gather_input_numel, world_size, dtype, device
        )  # no-op after 1st call
        fsdp_param.alloc_all_gather_output()
        fsdp_param.init_unsharded_param()  # no-op after 1st call. Need to call here so that ._unsharded_param access below doesn't fail.
    all_gather_output = all_gather_output.view(world_size, -1)
    # NOTE: This is the biggest difference between eager and compile code path.
    # In eager, we directly copy from `all_gather_output` into `fsdp_param.all_gather_output` (`fsdp_param._unsharded_param` will be updated because of shared storage),
    # but in compile path we copy from `as_strided(all_gather_output)` into `fsdp_param._unsharded_param` to avoid having `fsdp_param.all_gather_output` as graph input.
    # They are equivalent and must produce the same result.
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        out = [
            fsdp_param.all_gather_output.view(world_size, -1) for fsdp_param in fsdp_params
        ]
        torch.split_with_sizes_copy(
            all_gather_output, all_gather_input_numels, dim=1, out=out
        )
    else:
        splits = torch.split(all_gather_output, all_gather_input_numels, dim=1)
        out = []
        splits_unpadded = []
        assert len(fsdp_params) == len(splits)
        for i, fsdp_param in enumerate(fsdp_params):
            unsharded_param = fsdp_param._unsharded_param
            if fsdp_param.is_dtensor:
                unsharded_param = unsharded_param.to_local()
            out.append(unsharded_param)
            splits_unpadded.append(
                torch.as_strided(
                    splits[i].contiguous().view(splits[i].numel()),
                    fsdp_param._orig_size,
                    fsdp_param._contiguous_orig_stride,
                    storage_offset=0,
                )
            )
        ctx = contextlib.nullcontext()
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            ctx = torch.autograd._unsafe_preserve_version_counter_for_tensors(out)
        with torch.no_grad(), ctx:
            torch._foreach_copy_(out, splits_unpadded)


@torch.no_grad()
def foreach_reduce(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.cuda.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
    all_reduce_group: Optional[dist.ProcessGroup],
    all_reduce_stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    predivide_factor, postdivide_factor = divide_factors
    world_size = reduce_scatter_group.size()
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    ctx = contextlib.nullcontext()
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        current_stream = torch.cuda.current_stream()
        reduce_scatter_stream.wait_stream(current_stream)
        ctx = torch.cuda.stream(reduce_scatter_stream)
    with ctx:
        reduce_scatter_input = torch.empty(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )
        foreach_reduce_scatter_copy_in(
            unsharded_grads, reduce_scatter_input, world_size
        )
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # Only after the copy-in finishes can we free the gradients, which were
            # computed in the default stream
            current_stream.wait_stream(reduce_scatter_stream)
        unsharded_grads.clear()
        post_reduce_output = reduce_scatter_input.new_empty(
            (reduce_scatter_output_numel,)
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        _reduce_scatter(
            post_reduce_output,
            reduce_scatter_input,
            reduce_scatter_group,
            divide_factors,
        )
    view_out_stream = reduce_scatter_stream
    if all_reduce_group is not None:
        view_out_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with torch.cuda.stream(all_reduce_stream):
            _all_reduce(post_reduce_output, all_reduce_group, divide_factors)
    with torch.cuda.stream(view_out_stream):
        _div_if_needed(post_reduce_output, postdivide_factor)
        post_reduce_output = _to_dtype_if_needed(post_reduce_output, orig_dtype)
        # - View out and accumulate
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            new_sharded_grad = torch.as_strided(
                post_reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            if to_accumulate_grad:
                fsdp_param.sharded_param.grad += new_sharded_dtensor_grad
            else:
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            post_reduce_view_out_event = view_out_stream.record_event()
        else:
            post_reduce_view_out_event = None
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return post_reduce_view_out_event


def foreach_reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    grad_views: List[torch.Tensor] = []
    grads_to_copy: List[torch.Tensor] = []
    padded_grad_slices: List[torch.Tensor] = []
    for grad in unsharded_grads:
        grad_size = grad.size()
        dim0_padded_size = _get_dim0_padded_size(grad_size, world_size)
        if dim0_padded_size != grad_size:
            padded_grad = grad.new_empty(dim0_padded_size)
            padded_grad_slices.append(padded_grad[: grad.size(0)])
            grads_to_copy.append(grad)
            grad = padded_grad
        grad_views.append(grad.view(world_size, -1))
    if padded_grad_slices:
        with torch.no_grad():
            torch._foreach_copy_(padded_grad_slices, grads_to_copy)
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        torch.cat(grad_views, dim=-1, out=reduce_scatter_input.view(world_size, -1))
    else:
        cat_out = torch.cat(grad_views, dim=-1)
        reduce_scatter_input_view = reduce_scatter_input.view(world_size, -1)
        with torch.no_grad():
            reduce_scatter_input_view.copy_(cat_out)


def _reduce_scatter(
    output: torch.Tensor,
    input: torch.Tensor,
    group: dist.ProcessGroup,
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
) -> None:
    if divide_factors[0]:
        dist.reduce_scatter_tensor(output, input, group=group)
    else:
        # Using NCCL's reduce-scatter to do the division by world size saves
        # extra memory read/write from a separate division kernel
        dist.reduce_scatter_tensor(output, input, op=ReduceOp.AVG, group=group)


def _all_reduce(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    divide_factors: Union[Tuple[None, None], Tuple[float, float]],
) -> None:
    if divide_factors[0]:
        dist.all_reduce(tensor, group=group)
    else:
        # saves extra memory read/write from a separate division kernel
        dist.all_reduce(tensor, op=ReduceOp.AVG, group=group)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
