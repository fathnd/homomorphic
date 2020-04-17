import torch
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator
from .sgd import SGD
from ._amp_helper import _combined_found_inf_helper


class SRSGD(SGD):
    r"""Implements stochastic gradient descent with Stochastic Rounding.

    With Stochastic Rounding, param and `momentum_buffer` can be represented with 16 bits.
    See :func:`torch.stochastic_rounding` for details. This optimizer requires CUDA.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    _step_supports_amp_scaling = True

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grad_scaler (:class:`torch.cuda.amp.GradScaler`, optional):
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if grad_scaler is not None:
            inv_scale = grad_scaler._get_scale_async().double().reciprocal().float()
            found_inf = _combined_found_inf_helper(self, grad_scaler, inv_scale.device)
        else:
            inv_scale = torch.ones((1,), dtype=torch.float, device=torch.cuda.current_device())
            found_inf = _MultiDeviceReplicator(
                torch.zeros((1,), dtype=torch.float, device=torch.cuda.current_device()))

        inv_scale = _MultiDeviceReplicator(inv_scale)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError('SRSGD does not support sparse gradients')

                first_run = False
                param_state = self.state[param]
                if 'momentum_buffer' not in param_state:
                    first_run = True
                    param_state['momentum_buffer'] = torch.zeros_like(param)
                momentum_buffer = param_state['momentum_buffer']

                torch.stochastic_rounding_sgd_step(
                    param, grad, momentum_buffer,
                    inv_scale.get(param.device), found_inf.get(param.device),
                    group['lr'], momentum, weight_decay, dampening,
                    nesterov, first_run)

        return loss
