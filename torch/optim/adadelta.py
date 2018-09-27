import torch

from .optimizer import Optimizer


class Adadelta(Optimizer):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay factor (default: 0)
        l2_reg (boolean, optional): whether to use the original L2
            weight regularization or the weight decay method from the paper
            `Fixing Weight Decay Regularization in Adam` (default: True)

    __ https://arxiv.org/abs/1212.5701
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, l2_reg=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, l2_reg=l2_reg)
        super(Adadelta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adadelta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('l2_reg', True)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0 and group['l2_reg']:
                    grad.add_(group['weight_decay'], p.data)

                square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)

                if group['weight_decay'] != 0 and not group['l2_reg']:
                    p.data.add_(-group['weight_decay'], p.data)

                p.data.add_(-group['lr'], delta)

                acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

        return loss
