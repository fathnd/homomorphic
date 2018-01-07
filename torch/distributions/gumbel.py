from numbers import Number

import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Gumbel(Distribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> m = Gumbel(torch.Tensor([1.0]), torch.Tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
         1.0124
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): Location parameter of the distribution
        scale (float or Tensor or Variable): Scale parameter of the distribution
    """
    has_rsample = True
    params = {'loc': constraints.positive, 'scale': constraints.real}

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super(Gumbel, self).__init__(batch_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.real

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # uniform_() is [0, 1), but we need (0, 1]
        uni_dist = 1 - self.scale.new(shape).uniform_()
        # X ~ Uniform(0, 1)
        # Y = loc - scale * ln (ln (X)) ~ Gumbel(loc, scale)
        return self.loc - self.scale * torch.log(-uni_dist.log())

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        z = (value - self.loc) / self.scale
        return -(self.scale.log() + z + torch.exp(-z))

    def entropy(self):
        return self.scale.log() + 1 + 0.5771
