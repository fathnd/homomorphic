import torch
from torch._C import _ImperativeEngine as ImperativeEngine


__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()


compiled_autograd_final_callbacks = []


def queue_callback(cb):
    global compiled_autograd_final_callbacks
    compiled_autograd_final_callbacks.append(cb)
