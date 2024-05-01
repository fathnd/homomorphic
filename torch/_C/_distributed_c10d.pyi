# mypy: disable-error-code="type-arg"
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, overload, Tuple, Union

import torch
from torch import Tensor
from torch._C import ScriptObject
from torch.futures import Future

# This module is defined in torch/csrc/distributed/c10d/init.cpp

_DEFAULT_FIRST_BUCKET_BYTES: int
_DEFAULT_NO_TIMEOUT: timedelta
_DEFAULT_PG_TIMEOUT: timedelta
_DEFAULT_PG_NCCL_TIMEOUT: timedelta

class BuiltinCommHookType(Enum):
    ALLREDUCE = ...
    FP16_COMPRESS = ...

def _register_comm_hook(reducer: Reducer, state: Any, comm_hook: Any): ...
def _register_builtin_comm_hook(
    reducer: Reducer,
    comm_hook_type: BuiltinCommHookType,
): ...
def _set_global_rank(rank: int) -> None: ...
def _hash_tensors(tensors: List[Tensor]) -> int: ...

class GradBucket:
    def index(self) -> int: ...
    def buffer(self) -> Tensor: ...
    def gradients(self) -> List[Tensor]: ...
    def is_last(self) -> bool: ...
    def set_buffer(self, tensor: Tensor) -> None: ...
    def parameters(self) -> List[Tensor]: ...

class Reducer:
    def __init__(
        self,
        params: List[Tensor],
        bucket_indices: List[List[int]],
        per_bucket_size_limits: List[int],
        process_group: ProcessGroup,
        expect_sparse_gradients: List[bool] = ...,
        bucket_bytes_cap: int = ...,  # kDefaultBucketBytesCap in reducer.hpp
        find_unused_parameters: bool = ...,
        gradient_as_bucket_view: bool = ...,
        param_to_name_mapping: Dict[int, str] = ...,
        first_bucket_types_cap: int = ...,  # kDefaultFirstBucketBytes in reducer.hpp
    ): ...
    def prepare_for_forward(self) -> None: ...
    def prepare_for_backward(self, output: List[Tensor]) -> None: ...
    def get_backward_stats(self) -> List[int]: ...
    def _install_post_backward_futures(self, futures: List[Future]) -> None: ...
    def _rebuild_buckets(self) -> bool: ...
    def _get_zeros_like_grad_buckets(self) -> List[GradBucket]: ...
    def _push_all_rebuilt_params(self) -> None: ...
    def _set_forward_pass_work_handle(
        self,
        work: Work,
        use_static_world_size: bool,
    ): ...
    def _get_local_used_map(self) -> Tensor: ...
    def _set_ddp_runtime_logging_sample_rate(self, sample_rate: int) -> None: ...
    def _set_static_graph(self) -> None: ...
    def _run_comm_hook(self, bucket: GradBucket) -> Future: ...
    def set_logger(self, logger: Logger) -> None: ...
    def _remove_autograd_hooks(self) -> None: ...
    def _check_reducer_finalized(self) -> None: ...
    def _set_sparse_metadata(self, global_unique_ids: Dict[str, Tensor]) -> None: ...
    def _reset_state(self) -> None: ...
    def _update_process_group(self, new_process_group: ProcessGroup) -> None: ...

class DDPLoggingData:
    strs_map: Dict[str, str]
    ints_map: Dict[str, int]

class Logger:
    def __init__(self, reducer: Reducer): ...
    def set_construction_data_and_log(
        self,
        module_name: str,
        device_ids: List[int],
        output_device: int,
        broadcast_buffers: bool,
        has_sync_bn: bool,
        static_graph: bool,
    ): ...
    def set_runtime_stats_and_log(self) -> None: ...
    def set_error_and_log(self, error: str) -> None: ...
    def _get_ddp_logging_data(self) -> DDPLoggingData: ...
    def _set_comm_hook_name(self, comm_hook: str) -> None: ...
    def _set_uneven_input_join(self) -> None: ...
    def _set_static_graph(self) -> None: ...

def get_debug_level(): ...
def set_debug_level(): ...
def set_debug_level_from_env(): ...

class DebugLevel(Enum):
    OFF = ...
    INFO = ...
    DETAIL = ...

class ReduceOp:
    def __init__(self, op: RedOpType): ...

    SUM: RedOpType = ...
    AVG: RedOpType = ...
    PRODUCT: RedOpType = ...
    MIN: RedOpType = ...
    MAX: RedOpType = ...
    BAND: RedOpType = ...
    BOR: RedOpType = ...
    BXOR: RedOpType = ...
    PREMUL_SUM: RedOpType = ...
    UNUSED: RedOpType = ...

    class RedOpType(Enum): ...

class BroadcastOptions:
    rootRank: int
    rootTensor: int
    timeout: timedelta
    asyncOp: bool

class AllreduceOptions:
    reduceOp: ReduceOp
    timeout: timedelta

class AllreduceCoalescedOptions(AllreduceOptions): ...

class ReduceOptions:
    reduceOp: ReduceOp
    rootRank: int
    rootTensor: int
    timeout: timedelta

class AllgatherOptions:
    timeout: timedelta
    asyncOp: bool

class GatherOptions:
    rootRank: int
    timeout: timedelta

class ScatterOptions:
    rootRank: int
    timeout: timedelta
    asyncOp: bool

class ReduceScatterOptions:
    reduceOp: ReduceOp
    timeout: timedelta
    asyncOp: bool

class BarrierOptions:
    device_ids: List[int]
    device: torch.device
    timeout: timedelta

class AllToAllOptions:
    timeout: timedelta

class Store:
    def set(self, key: str, value: str): ...
    def get(self, key: str) -> bytes: ...
    def add(self, key: str, value: int) -> int: ...
    def compare_set(
        self,
        key: str,
        expected_value: str,
        desired_value: str,
    ) -> bytes: ...
    def delete_key(self, key: str) -> bool: ...
    def num_keys(self) -> int: ...
    def set_timeout(self, timeout: timedelta): ...
    @overload
    def wait(self, keys: List[str]): ...
    @overload
    def wait(self, keys: List[str], timeout: timedelta): ...

class FileStore(Store):
    def __init__(self, path: str, numWorkers: int = ...): ...

class HashStore(Store):
    def __init__(self): ...

class TCPStore(Store):
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: Optional[int] = ...,
        is_master: bool = ...,
        timeout: timedelta = ...,
        wait_for_workers: bool = ...,
        multi_tenant: bool = ...,
        master_listen_fd: Optional[int] = ...,
        use_libuv: Optional[bool] = ...,
    ): ...
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...

class PrefixStore(Store):
    def __init__(self, prefix: str, store: Store): ...
    @property
    def underlying_store(self) -> Store: ...

class _DistributedBackendOptions:
    def __init__(self): ...
    @property
    def store(self) -> Store: ...
    @store.setter
    def store(self, store: Store) -> None: ...
    @property
    def group_rank(self) -> int: ...
    @group_rank.setter
    def group_rank(self, rank: int) -> None: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(self, size: int) -> None: ...
    @property
    def timeout(self) -> timedelta: ...
    @timeout.setter
    def timeout(self, timeout: timedelta) -> None: ...
    @property
    def group_id(self) -> str: ...
    @group_id.setter
    def group_id(self, group_id: str) -> None: ...
    @property
    def global_ranks_in_group(self) -> List[int]: ...
    @global_ranks_in_group.setter
    def global_ranks_in_group(self, ranks: List[int]) -> None: ...

class Work:
    def is_completed(self) -> bool: ...
    def is_success(self) -> bool: ...
    def exception(self) -> Any: ...
    def wait(self, timeout: timedelta = ...) -> bool: ...
    def get_future(self) -> Future: ...
    def source_rank(self) -> int: ...
    def _source_rank(self) -> int: ...
    def result(self) -> List[Tensor]: ...
    def synchronize(self): ...
    def boxed(self) -> ScriptObject: ...
    @staticmethod
    def unbox(obj: ScriptObject) -> Work: ...

class Backend:
    def __init__(
        self,
        rank: int,
        size: int,
    ): ...
    @property
    def supports_splitting(self) -> bool: ...
    def rank(self) -> int: ...
    def size(self) -> int: ...
    def eager_connect_single_device(self, device: Optional[torch.device]) -> None: ...
    def _set_sequence_number_for_group(self) -> None: ...

class ProcessGroup:
    class Options:
        def __init__(self, backend: str, timeout: timedelta = ...): ...
        @property
        def backend(self) -> str: ...
        @property
        def _timeout(self) -> timedelta: ...
        @_timeout.setter
        def _timeout(self, val: timedelta) -> None: ...

    class BackendType(Enum):
        UNDEFINED = ...
        GLOO = ...
        NCCL = ...
        UCC = ...
        MPI = ...
        CUSTOM = ...
    def __init__(self, store: Store, rank: int, size: int, options: Options): ...
    def rank(self) -> int: ...
    def size(self) -> int: ...
    @overload
    def broadcast(
        self,
        tensors: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def broadcast(
        self,
        tensor: Tensor,
        root: int,
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensors: List[Tensor],
        opts: AllreduceOptions = ...,
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensors: List[Tensor],
        op=...,
    ) -> Work: ...
    @overload
    def allreduce(
        self,
        tensor: Tensor,
        op=...,
    ) -> Work: ...
    def allreduce_coalesced(
        self,
        tensors: List[Tensor],
        opts=...,
    ) -> Work: ...
    def reduce_scatter_tensor_coalesced(
        self,
        outputTensors: List[Tensor],
        inputTensors: List[Tensor],
        opts: Optional[ReduceScatterOptions] = None,
    ) -> Work: ...
    @overload
    def reduce(
        self,
        tensors: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def reduce(
        self,
        tensor: Tensor,
        root: int,
        op=...,
    ) -> Work: ...
    @overload
    def allgather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def allgather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
    ) -> Work: ...
    def _allgather_base(
        self,
        output: Tensor,
        input: Tensor,
        opts=...,
    ) -> Work: ...
    def allgather_coalesced(
        self,
        output_lists: List[List[Tensor]],
        input_list: List[Tensor],
        opts=...,
    ) -> Work: ...
    def allgather_into_tensor_coalesced(
        self,
        output_lists: List[Tensor],
        input_list: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def gather(
        self,
        output_tensors: List[List[Tensor]],
        input_tensors: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def gather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
        root: int,
    ) -> Work: ...
    @overload
    def scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts=...,
    ) -> Work: ...
    @overload
    def scatter(
        self,
        output_tensor: Tensor,
        input_tensors: List[Tensor],
        root: int,
    ) -> Work: ...
    @overload
    def reduce_scatter(
        self,
        output_tensors: List[Tensor],
        input_tensors: List[List[Tensor]],
        opts=...,
    ) -> Work: ...
    @overload
    def reduce_scatter(
        self,
        output_tensors: Tensor,
        input_tensor: List[Tensor],
    ) -> Work: ...
    def _reduce_scatter_base(
        self,
        outputTensor: Tensor,
        inputTensor: Tensor,
        opts: Optional[ReduceScatterOptions],
    ) -> Work: ...
    @overload
    def alltoall_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts=...,
    ) -> Work: ...
    @overload
    def alltoall_base(
        self,
        output: Tensor,
        input: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
    ) -> Work: ...
    @overload
    def alltoall(
        self,
        output_tensor: List[Tensor],
        input_tensor: List[Tensor],
        opts=...,
    ) -> Work: ...
    @overload
    def alltoall(
        self,
        output: List[Tensor],
        input: List[Tensor],
    ) -> Work: ...
    def send(
        self,
        tensors: List[Tensor],
        dstRank: int,
        tag: int,
    ) -> Work: ...
    def recv(
        self,
        tensors: List[Tensor],
        srcRank: int,
        tag: int,
    ) -> Work: ...
    def recv_anysource(self, tensors: List[Tensor], tag: int) -> Work: ...
    def barrier(self, opts=...) -> Work: ...
    def boxed(self) -> ScriptObject: ...
    @staticmethod
    def unbox(obj: ScriptObject) -> ProcessGroup: ...
    def _start_coalescing(self, device: torch.device) -> None: ...
    def _end_coalescing(self, device: torch.device) -> Work: ...
    def _get_backend_name(self) -> str: ...
    def _backend_id(self, backend_type: BackendType) -> int: ...
    @property
    def _device_types(self) -> List[torch.device]: ...
    def _get_backend(self, device: torch.device) -> Backend: ...
    def _register_backend(
        self,
        device: torch.device,
        backend_type: BackendType,
        backend: Optional[Backend],
    ) -> None: ...
    def _set_group_name(self, name: str) -> None: ...
    def _set_group_desc(self, desc: str) -> None: ...
    def name(self) -> str: ...
    def _has_hooks(self) -> bool: ...
    def _wait_for_pending_works(self) -> None: ...
    def _set_sequence_number_for_group(self) -> None: ...
    @property
    def bound_device_id(self) -> Optional[torch.device]: ...
    @bound_device_id.setter
    def bound_device_id(self, device: Optional[torch.device]) -> None: ...
    @property
    def group_name(self) -> str: ...
    @property
    def group_desc(self) -> str: ...

class ProcessGroupRoundRobin(ProcessGroup): ...

def _round_robin_process_groups(
    process_groups: List[ProcessGroup],
) -> ProcessGroupRoundRobin: ...

class ProcessGroupGloo(Backend):
    class Device: ...
    class Options: ...

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
    @staticmethod
    def create_device(hostname="", interface="") -> Device: ...
    @staticmethod
    def create_default_device() -> Device: ...
    def _set_default_timeout(self, timeout) -> None: ...

class _ProcessGroupWrapper(Backend):
    def __init__(self, pg: Backend, gloo_pg: ProcessGroupGloo): ...
    wrapped_pg: Backend

class ProcessGroupNCCL(Backend):
    class Options:
        def __init__(self, timeout: Optional[timedelta] = None): ...
        @property
        def backend(self) -> str: ...
        @property
        def _timeout(self) -> timedelta: ...
        @_timeout.setter
        def _timeout(self, val: timedelta) -> None: ...
        @property
        def _is_high_priority_stream(self) -> bool: ...
        @_is_high_priority_stream.setter
        def _is_high_priority_stream(self, val: bool) -> None: ...

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...
    def _group_start(self) -> None: ...
    def _group_end(self) -> None: ...
    def _set_default_timeout(self, timeout) -> None: ...
    def _shutdown(self) -> None: ...
    @property
    def uid(self) -> int: ...

class ProcessGroupUCC(Backend):
    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        timeout: timedelta,
    ): ...

class ProcessGroupMPI(Backend):
    def __init__(
        self,
        rank: int,
        size: int,
        pgComm: int,
    ): ...
    @staticmethod
    def create(ranks: List[int]) -> ProcessGroupMPI: ...

def _compute_bucket_assignment_by_size(
    tensors: List[Tensor],
    bucket_size_limits: List[int],
    expect_sparse_gradient: List[bool] = ...,
    tensor_indices: List[int] = ...,
) -> Tuple[List[List[int]], List[int]]: ...
def _broadcast_coalesced(
    process_group: ProcessGroup,
    tensors: List[Tensor],
    buffer_size: int,
    src: int,
): ...
def _test_python_store(store: Store): ...
def _verify_params_across_processes(
    process_group: ProcessGroup,
    params: List[Tensor],
    logger: Optional[Logger],
): ...
def _make_nccl_premul_sum(factor: Union[float, List[Tensor]]) -> ReduceOp: ...
def _register_process_group(
    group_name: str,
    process_group: ProcessGroup,
) -> None: ...
def _resolve_process_group(group_name: str) -> ProcessGroup: ...
def _unregister_all_process_groups() -> None: ...
def _unregister_process_group(group_name: str) -> None: ...

class ProcessGroupCudaP2P(Backend):
    class Options:
        nccl_options: Optional[ProcessGroupNCCL.Options]
        buffer_size: Optional[int]

        def __init__(self) -> None: ...

    def __init__(
        self,
        store: Store,
        rank: int,
        size: int,
        options: ProcessGroupCudaP2P.Options,
    ) -> None: ...
    def is_p2p_available(self) -> bool: ...
    def _shutdown(self) -> None: ...
