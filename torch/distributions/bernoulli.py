from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs


class Bernoulli(Distribution):
    r"""
    Creates a Bernoulli distribution parameterized by `probs`.

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> m = Bernoulli(torch.Tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
         0.0
        [torch.FloatTensor of size 1]

    Args:
        probs (Number, Tensor or Variable): the probabilty of sampling `1`
    """
    params = {'probs': constraints.unit_interval}
    support = constraints.boolean
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None):
        if (probs is None) == (logits is None):
            raise ValueError("Got probs={}, logits={}. Either `probs` or `logits` must be specified, "
                             "but not both.".format(probs, logits))
        if probs is not None:
            self.probs, = broadcast_all(probs)
            self.logits = probs_to_logits(self.probs, is_binary=True)
        else:
            self.logits, = broadcast_all(logits)
            self.probs = logits_to_probs(self.logits, is_binary=True)
        if isinstance(probs, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.probs.size()
        super(Bernoulli, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.bernoulli(self.probs.expand(shape))

    def _binary_cross_entropy(self, value):
        max_val = (-self.logits).clamp(min=0)
        return self.logits - self.logits * value + max_val + \
            ((-max_val).exp() + (-self.logits - max_val).exp()).log()

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return -self._binary_cross_entropy(value)

    def entropy(self):
        return self._binary_cross_entropy(self.probs)

    def enumerate_support(self):
        values = torch.arange(2).long()
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
