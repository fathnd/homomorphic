# Owner(s): ["module: dynamo"]

"""Light smoke test switching between numpy to pytorch random streams.
"""
from contextlib import contextmanager

import numpy as _np
import pytest

import torch._numpy as tnp
from torch._numpy.testing import assert_equal


@contextmanager
def control_stream(use_numpy=False):
    oldstate = tnp.random.USE_NUMPY_RANDOM
    tnp.random.USE_NUMPY_RANDOM = use_numpy
    try:
        yield
    finally:
        tnp.random.USE_NUMPY_RANDOM = oldstate


@pytest.mark.parametrize("use_numpy", [True, False])
def test_uniform(use_numpy):
    with control_stream(use_numpy):
        r = tnp.random.uniform(0, 1, size=10)
    assert isinstance(r, tnp.ndarray)


@pytest.mark.parametrize("use_numpy", [True, False])
def test_uniform_scalar(use_numpy):
    # default `size` means a python scalar return
    with control_stream(use_numpy):
        r = tnp.random.uniform(0, 1)
    assert isinstance(r, float)


def test_shuffle():
    x = tnp.arange(10)
    tnp.random.shuffle(x)


@pytest.mark.parametrize("use_numpy", [True, False])
def test_choice(use_numpy):
    kwds = dict(size=3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    with control_stream(use_numpy):
        tnp.random.seed(12345)
        x = tnp.random.choice(5, **kwds)
        x_1 = tnp.random.choice(tnp.arange(5), **kwds)
        assert_equal(x, x_1)


def test_numpy_global():
    with control_stream(use_numpy=True):
        tnp.random.seed(12345)
        x = tnp.random.uniform(0, 1, size=11)

    # check that the stream is identical to numpy's
    _np.random.seed(12345)
    x_np = _np.random.uniform(0, 1, size=11)
    assert_equal(x, tnp.asarray(x_np))

    # switch to the pytorch stream, variates differ
    with control_stream(use_numpy=False):
        tnp.random.seed(12345)
        x_1 = tnp.random.uniform(0, 1, size=11)

    assert not (x_1 == x).all()


def test_wrong_global():
    with control_stream("oops"):
        with pytest.raises(ValueError):
            tnp.random.rand()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
