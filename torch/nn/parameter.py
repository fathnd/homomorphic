import enum
import torch
from collections import OrderedDict

class ParameterMode(enum.Enum):
    Infer = -1


class Parameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )


class _UninitializedParameter(Parameter):
    r"""A parameter that is not yet initialized for shape inference support.
    """
    def __new__(cls, requires_grad=True):
        data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def materialize(self, shape, device=None, dtype=None):
        r"""Given a shape, it materializes a parameter in the same device
        as the current one"""
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        return Parameter(torch.empty(shape, device=device, dtype=dtype))

    def __repr__(self):
        return 'Uninitialized parameter'


class _UninitializedBuffer(torch.Tensor):
    r"""A buffer that is not yet initialized for shape inference support.
    """
    def __new__(cls):
        data = torch.Tensor()
        requires_grad = False
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return 'Uninitialized buffer'
