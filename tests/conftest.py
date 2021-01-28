import pytest

import torch

from e3nn.util.equivariance import equivariance_error


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
    tolerances = {
        'float32': 1e-4,
        'float64': 1e-10
    }
    return torch.as_tensor(tolerances[request.param])


@pytest.fixture(scope='session')
def assert_equivariant(float_tolerance):
    r"""Assert that the ``equivariance_error`` is below the ``float_tolerance``.

    Parameters
    ----------
    func : callable
        the function to test
    sqrt_tolerance : bool
        whether to replace ``float_tolerance`` with ``sqrt(float_tolerance)``.
    tolerance_multiplier : float
        ``float_tolerance`` is replaced by ``tolerance_multiplier*float_tolerance``. Defaults to 1.
    **kwargs : kwargs
        passed through to ``equivariance_error``.

    Returns
    -------
    None
    """
    # TODO: record statistics on equivariance error
    def func(*args, sqrt_tolerance=False, tolerance_multiplier=1., **kwargs):
        # Apply a default:
        if 'ntrials' not in kwargs:
            kwargs['ntrials'] = 1
        error = equivariance_error(*args, **kwargs)
        tol = tolerance_multiplier*float_tolerance
        if sqrt_tolerance:
            tol = torch.sqrt(tol)
        assert error <= tol, "Largest componentwise equivariance error %f too large" % (error,)
    return func
