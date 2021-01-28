import pytest

import torch

from e3nn import o3
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
    # TODO: record statistics on equivariance error
    def func(*args, sqrt_tolerance=False, tolerance_multiplier=1., **kwargs):
        # Apply a default:
        if 'ntrials' not in kwargs:
            kwargs['ntrials'] = 2
        error = equivariance_error(*args, **kwargs)
        tol = tolerance_multiplier*float_tolerance
        if sqrt_tolerance:
            tol = torch.sqrt(tol)
        assert error <= tol, "Maximum componentwise equivariance error: %r" % error
    return func


@pytest.fixture(scope='session')
def assert_model_equivariant(assert_equivariant):
    from torch_geometric.data import Data

    def func(model, n_nodes=(5, 10), **kwargs):
        assert 'irreps_in' not in kwargs
        assert 'irreps_out' not in kwargs
        assert 'batch_dim' not in kwargs

        if isinstance(n_nodes, int):
            n_nodes = (n_nodes, n_nodes)

        assert hasattr(model, 'irreps_out')
        has_features = getattr(model, 'irreps_in', None) is not None
        has_attrs = getattr(model, 'irreps_node_attr', None) is not None
        assert has_features or has_attrs

        irreps_in = [o3.Irreps('1x1e')]  # for pos
        irreps_in += ([model.irreps_in] if has_features else []) 
        irreps_in += ([model.irreps_node_attr] if has_attrs else [])

        def wrapper(*args):
            args = list(reversed(args))
            assert len(args) == len(irreps_in)
            fields = {}
            fields['pos'] = args.pop()
            print(fields['pos'].shape)
            nnodes = fields['pos'].shape[0]
            if has_features:
                fields['x'] = args.pop()
                assert fields['x'].shape[0] == nnodes
            if has_attrs:
                fields['z'] = args.pop()
                assert fields['z'].shape[0] == nnodes
            fields['batch'] = torch.zeros(nnodes, dtype=torch.long)
            data = Data.from_dict(fields)
            print('data', data)
            out = model(data)
            print('out', out.shape)
            return out

        assert_equivariant(
            wrapper,
            irreps_in=irreps_in,
            irreps_out=[model.irreps_out],
            batch_dim=n_nodes,  # the batch is the number of nodes
            **kwargs
        )

    return func
