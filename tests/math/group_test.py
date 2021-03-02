import pytest

import itertools

import torch
from e3nn.math.group import O3, SO3, Sn, is_representation, is_group


def test_representation():
    if torch.get_default_dtype() != torch.float64:
        pytest.xfail('representations only check out with double accuracy')
    for group in [SO3(), O3(), Sn(5)]:
        for r in itertools.islice(group.irrep_indices(), 10):
            assert is_representation(group, group.rep(r), 1e-9)


def test_is_group():
    if torch.get_default_dtype() != torch.float64:
        pytest.xfail('groups only check out with double accuracy')
    assert is_group(O3(), 1e-3)
    assert is_group(SO3(), 1e-3)
