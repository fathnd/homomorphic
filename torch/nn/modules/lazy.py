import itertools
from typing_extensions import Protocol
import warnings

import torch
from ..parameter import UninitializedParameter


class _LazyProtocol(Protocol):
    """This is to avoid errors with mypy checks for 
    The attributes in a mixin:
    https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
    """
    def _register_load_state_dict_pre_hook(self, hook):
        ...

    def register_forward_pre_hook(self, hook):
        ...

    def _lazy_load_hook(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        ...

    def _get_name(self):
        ...

    def _infer_parameters(self, module, input):
        ...

    @property
    def _parameters(self):
        ...

    @property
    def _buffers(self):
        ...

    @property
    def _non_persistent_buffers_set(self):
        ...

    @property
    def _load_hook(self):
        ...

    @property
    def _initialize_hook(self):
        ...


class LazyModuleMixin:
    r"""A mixin for modules that lazily initialize parameters, also known as "lazy modules."

    .. warning:
    Lazy modules are an experimental new feature under active development,
    and their API is likely to change.

    Modules that lazily initialize parameters, or "lazy modules",
    derive the shapes of their parameters from the first input(s)
    to their forward method. Until that first forward they contain
    :class:`torch.nn.UninitializedParameter`s that should not be accessed
    or used, and afterward they contain regular :class:`torch.nn.Parameter`s.
    Lazy modules are convenient since they don't require computing some
    module arguments, like the `in_features` argument of a
    typical :class:`torch.nn.Linear`.

    After construction, networks with lazy modules should first
    be converted to the desired dtype and placed on the desired device.
    The lazy modules should then be initialized with one or more "dry runs".
    These "dry runs" send inputs of the correct size, dtype, and device through
    the network and to each one of its lazy modules. After this the network can be used as usual.

    >>> class LazyMLP(torch.nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.fc1 = torch.nn.LazyLinear(10)
    ...        self.relu1 = torch.nn.ReLU()
    ...        self.fc2 = torch.nn.LazyLinear(1)
    ...        self.relu2 = torch.nn.ReLU()
    ...
    ...    def forward(self, input):
    ...        x = self.relu1(self.fc1(input))
    ...        y = self.relu2(self.fc2(x))
    ...        return y
    >>> # constructs a network with lazy modules
    >>> lazy_mlp = LazyMLP()
    >>> # transforms the network's device and dtype
    >>> # NOTE: these transforms can and should be applied after construction and before any 'dry runs'
    >>> lazy_mlp = mlp.cuda().double()
    >>> lazy_mlp
    LazyMLP(
      (fc1): LazyLinear(in_features=0, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): LazyLinear(in_features=0, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # performs a dry run to initialize the network's lazy modules
    >>> lazy_mlp(torch.ones(10,10).cuda())
    >>> # after initialization, LazyLinear modules become regular Linear modules
    >>> lazy_mlp
    LazyMLP(
      (fc1): Linear(in_features=10, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=10, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # attaches an optimizer, since parameters can now be used as usual
    >>> optim = torch.optim.SGD(mlp.parameters(), lr=0.01)

    Note that lazy modules cannot validate if the shape is correct during
    deserialization when loading an initialized parameter into an uninitialized one.

    A final caveat when using lazy modules is that the order of initialization of a network's
    parameters may change, since the lazy modules are always initialized after other modules.
    This can cause the parameters of a network using lazy modules to be initialized differently
    than the parameters of a network without lazy modules.
    For example, if the LazyMLP class defined above had a :class:`torch.nn.LazyLinear` module
    first and then a regular :class:`torch.nn.Linear` second, the second module would be
    initialized on construction and the first module would be initialized during the first dry run."

    .. note:: A `LazyModule`, or a module containing a `LazyModule` requires an explicit
    dummy forward call in the following scenarios.

    * Access an :class:`UnitializedParameter` data or properties
    >>> lazy_module.weight.shape

    * Use an Optimizer that accesses Parameters data in their constructor 
    >>> optim = torch.optim.AdaGrad(lazy_module.parameters())

    * Use :class:`torch.nn.DataParallel` or :class:`torch.nn.DistributedDataParallel`
    >>> module_dp = torch.nn.DataParallel(lazy_module)

    * Module reparametrization using :func:`torch.nn.utils.weight_norm` or :func:`torch.nn.utils.spectral_norm`
    >>> torch.nn.utils.weight_norm(lazy_module)

    * When dealing with non-default module initialization
    >>> torch.nn.init.uniform_(lazy_module.weight)

    """

    # modules inheriting from this will change their __class__ to the specified
    # one after they are fully initialized
    cls_to_become = None

    def __init__(self: _LazyProtocol, *args, **kwargs):
        # Mypy doesnt like this super call in a mixin
        super().__init__(*args, **kwargs)  # type: ignore
        self._load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        self._initialize_hook = self.register_forward_pre_hook(self._infer_parameters)
        warnings.warn('Lazy modules are a new feature under heavy development '
                      'so changes to the API or functionality can happen at any moment.')

    def _save_to_state_dict(self: _LazyProtocol, destination, prefix, keep_vars):
        # This should be ideally implemented as a hook, 
        # but we should override `detach` in the UninitializedParameter to return itself
        # which is not clean
        for name, param in self._parameters.items():
            if param is not None:
                if isinstance(param, UninitializedParameter):
                    destination[prefix + name] = param
                else:
                    destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    def _lazy_load_hook(
            self: _LazyProtocol, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        local_state = {k: v for k, v in self._parameters.items() if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if isinstance(param, UninitializedParameter): 
                    # The current parameter is not initialized but the one being loaded one is
                    # create a new parameter based on the uninitialized one
                    if not isinstance(input_param, UninitializedParameter):
                        with torch.no_grad():
                            param.materialize(input_param.shape)

    def initialize_parameters(self: _LazyProtocol, *args, **kwargs):
        r"""Initialize parameters according to the input batch properties.
        This adds an interface to isolate parameter initialization from the
        forward pass when doing parameter shape inference.
        """
        raise NotImplementedError('initialize_parameters is not implemented for {}'.format(self.__class__.__name__))

    def has_uninitialized_params(self: _LazyProtocol):
        r"""Check if a module has parameters that are not initialized
        """
        # This is to avoid the JIT to track this parameter and force
        # custom modules __setstate__ to add it
        params = self._parameters.values()
        for param in itertools.chain(params):
            if isinstance(param, (UninitializedParameter)):
                return True
        return False

    def _infer_parameters(self: _LazyProtocol, module, input):
        r"""Infers the size and initializes the parameters according to the
        provided input batch.
        Given a module that contains parameters that were declared inferrable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
        module.initialize_parameters(*input) 
        if module.has_uninitialized_params():
            raise RuntimeError('module {} has not been fully initialized'.format(self._get_name()))
        module._initialize_hook.remove()
        module._load_hook.remove()
        delattr(module, '_initialize_hook')
        delattr(module, '_load_hook')
        if module.cls_to_become is not None:
            module.__class__ = module.cls_to_become


    def _replicate_for_data_parallel(self: _LazyProtocol):
        raise RuntimeError('Modules with uninitialized parameters can\'t be used with `DataParallel`. '
                           'Run a dummy forward pass to correctly initialize the modules')                    
