import pytest

import torch

import e3nn.util.test


@pytest.fixture(scope='session', autouse=True, params=['float32', 'float64'])
def float_tolerance(request):
    """Run all tests with various default floating dtypes.

    Returns
    --------
        A precision threshold to use for closeness tests.
    """
    dtype = {
        'float32': torch.float32,
        'float64': torch.float64
    }[request.param]
    torch.set_default_dtype(dtype)
    return e3nn.util.test.EQUIVARIANCE_TOLERANCE[dtype]


@pytest.fixture(scope='session')
def assert_equivariant():
    return e3nn.util.test.assert_equivariant
