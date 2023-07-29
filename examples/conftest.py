import pytest

from e3nn.util import test


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set the random seeds to try to get some reproducibility"""
    test.set_random_seeds()
