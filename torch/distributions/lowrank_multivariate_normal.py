import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (_batch_diag, _batch_mahalanobis, _batch_mv,
                                                     _batch_potrf_lower, _batch_trtrs_lower,
                                                     _get_batch_shape)
from torch.distributions.utils import lazy_property


def _batch_vector_diag(bvec):
    """
    Returns the diagonal matrices of a batch of vectors.
    """
    n = bvec.shape[-1]
    flat_bvec = bvec.reshape(-1, n)
    flat_bmat = torch.stack([v.diag() for v in flat_bvec])
    return flat_bmat.reshape(bvec.shape + (n,))


def _batch_capacitance_tril(W, D):
    r"""
    Computes Cholesky of :math:`I + W.T @ inv(D) @ W` for a batch of matrices :math:`W`
    and a batch of vectors :math:`D`.
    """
    m = W.shape[-1]
    identity = torch.eye(m, out=W.new(m, m))
    Wt_Dinv = W.transpose(-1, -2) / D.unsqueeze(-2)
    K = identity + torch.matmul(Wt_Dinv, W)
    return _batch_potrf_lower(K)


def _batch_lowrank_logdet(W, D, capacitance_tril=None):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    if capacitance_tril is None:
        capacitance_tril = _batch_capacitance_tril(W, D)
    return 2 * _batch_diag(capacitance_tril).log().sum(-1) + D.log().sum(-1)


def _batch_lowrank_mahalanobis(W, D, x, capacitance_tril=None):
    r"""
    Uses "Woodbury matrix identity"::
        inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D),
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute the squared
    Mahalanobis distance :math:`x.T @ inv(W @ W.T + D) @ x`.
    """
    if capacitance_tril is None:
        capacitance_tril = _batch_capacitance_tril(W, D)
    Wt_Dinv = W.transpose(-1, -2) / D.unsqueeze(-2)
    Wt_Dinv_x = _batch_mv(Wt_Dinv, x)
    mahalanobis_term1 = (x.pow(2) / D).sum(-1)
    mahalanobis_term2 = _batch_mahalanobis(capacitance_tril, Wt_Dinv_x)
    return mahalanobis_term1 - mahalanobis_term2


class LowRankMultivariateNormal(Distribution):  # TODO: docs, tests
    r"""
    Creates a multivariate normal distribution with covariance matrix having a low-rank form
    parameterized by `scale_factor` and `scale_diag`::
        covariance_matrix = scale_factor @ scale_factor.T + scale_diag

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.tensor([1, 0]), torch.tensor([1, 1]))
        >>> m.sample()  # normally distributed with mean=`[0,0]`, scale_factor=`[1,0]`, scale_diag=`[1,1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        scale_factor (Tensor): factor part of low-rank form of covariance matrix
        scale_diag (Tensor): diagonal part of low-rank form of covariance matrix

    Note:
        The computation for determinant and inverse of covariance matrix is saved when
        `scale_factor.shape[1] << scale_factor.shape[0]` thanks to "Woodbury matrix identity" and
        "matrix determinant lemma". Thanks to these formulas, we just need to compute the
        determinant and inverse of the small size "capacitance" matrix::
            capacitance = I + scale_factor.T @ inv(scale_diag) @ scale_factor
    """
    arg_constraints = {"loc": constraints.real,
                       "scale_factor": constraints.real,
                       "scale_diag": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale_factor, scale_diag, validate_args=None):
        event_shape = loc.shape[-1:]
        if scale_factor.dim() < 2:
            raise ValueError("scale_factor must be at least two-dimensional, "
                             "with optional leading batch dimensions")
        if scale_factor.shape[-2:-1] != event_shape:
            raise ValueError("scale_factor must be a batch of matrices with shape {} x m"
                             .format(event_shape[0]))
        if scale_diag.shape[-1:] != event_shape:
            raise ValueError("scale_diag must be a batch of vectors with shape {}".format(event_shape))

        scale_batch_shape = _get_batch_shape(scale_factor, scale_diag)
        try:
            batch_shape = torch._C._infer_size(loc.shape[:-1], scale_batch_shape)
        except:
            raise ValueError("Incompatible batch shapes: loc {}, scale_factor {}, scale_diag {}"
                             .format(loc.shape, scale_factor.shape, scale_diag.shape))

        loc_shape = batch_shape + event_shape
        self.loc = loc.expand(loc_shape)
        self.scale_factor = scale_factor.expand(loc_shape + scale_factor.shape[-1:])
        self.scale_diag = scale_diag.expand(loc_shape)
        super(LowRankMultivariateNormal, self).__init__(batch_shape, event_shape,
                                                        validate_args=validate_args)
        self._capacitance_tril  = _batch_capacitance_tril(self.scale_factor, self.scale_diag)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale_factor.pow(2).sum(-1) + self.scale_diag

    @lazy_property
    def scale_tril(self):
        # The following identity is used to increase the numerically computation stability
        # for Cholesky decomposition (see http://www.gaussianprocess.org/gpml/, Section 3.4.3):
        #     W @ W.T + D = D1/2 @ (I + D-1/2 @ W @ W.T @ D-1/2) @ D1/2
        # The matrix "I + D-1/2 @ W @ W.T @ D-1/2" has eigenvalues bounded from below by 1,
        # hence it is well-conditioned and safe to take Cholesky decomposition.
        n = self._event_shape[0]
        identity = torch.eye(n, out=self.scale_factor.new(n, n))
        scale_diag_sqrt_unsqueeze = self.scale_diag.sqrt().unsqueeze(-1)
        Dinvsqrt_W = self.scale_factor / scale_diag_sqrt_unsqueeze
        K = identity + torch.matmul(Dinvsqrt_W, Dinvsqrt_W.transpose(-1, -2))
        return scale_diag_sqrt_unsqueeze * _batch_potrf_lower(K)

    @lazy_property
    def covariance_matrix(self):
        return (torch.matmul(self.scale_factor, self.scale_factor.transpose(-1, -2))
                + _batch_vector_diag(self.scale_diag))

    @lazy_property
    def precision_matrix(self):
        # We use "Woodbury matrix identity" to take advantage of low rank form::
        #     inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D)
        # where :math:`C` is the capacitance matrix.
        Wt_Dinv = self.scale_factor.transpose(-1, -2) / self.scale_diag.unsqueeze(-2)
        A = _batch_trtrs_lower(Wt_Dinv, self._capacitance_tril)
        return _batch_vector_diag(self.scale_diag.reciprocal()) - torch.matmul(A.transpose(-1, -2), A)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_lowrank_mahalanobis(self.scale_factor, self.scale_diag, diff,
                                       self._capacitance_tril)
        log_det = _batch_lowrank_logdet(self.scale_factor, self.scale_diag,
                                        self._capacitance_tril)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + log_det + M)

    def entropy(self):
        log_det = _batch_lowrank_logdet(self.scale_factor, self.scale_diag,
                                        self._capacitance_tril)
        H = 0.5 * (self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + log_det)
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
