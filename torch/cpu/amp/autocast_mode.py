import torch
from typing import Any

class autocast(torch.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``
    """
    def __init__(self, enabled : bool = True, dtype : torch.dtype = torch.bfloat16, cache_enabled : bool = True):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            return
        super().__init__("cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
