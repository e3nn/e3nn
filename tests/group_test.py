import itertools

import torch
from e3nn.math.group import O3, SO3, Sn, is_representation, is_group


def test_representation():
    torch.set_default_dtype(torch.float64)
    for group in [SO3(), O3(), Sn(5)]:
        for r in itertools.islice(group.irrep_indices(), 10):
            assert is_representation(group, group.rep(r), 1e-9)


def test_is_group():
    assert is_group(O3(), 1e-3)
    assert is_group(SO3(), 1e-3)
