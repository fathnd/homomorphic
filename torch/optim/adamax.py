import torch
from .optimizer import Optimizer


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm).

    It has been proposed in `Adam: A Method for Stochastic Optimization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay factor (default: 0)
        l2_reg (boolean, optional): whether to use the original L2
            weight regularization or the weight decay method from the paper
            `Fixing Weight Decay Regularization in Adam` (default: True)

    __ https://arxiv.org/abs/1412.6980
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, l2_reg=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        l2_reg=l2_reg)
        super(Adamax, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adamax, self).__setstate__(state)
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
                    raise RuntimeError('Adamax does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0 and group['l2_reg']:
                    grad.add_(group['weight_decay'], p.data)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

                bias_correction = 1 - beta1 ** state['step']
                clr = group['lr'] / bias_correction

                if group['weight_decay'] != 0 and not group['l2_reg']:
                    p.data.add_(-group['weight_decay'], p.data)

                p.data.addcdiv_(-clr, exp_avg, exp_inf)

        return loss
