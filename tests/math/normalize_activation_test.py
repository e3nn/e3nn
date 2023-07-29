import torch
from e3nn.math import normalize2mom


def test_device() -> None:
    act = torch.nn.ReLU()
    act = normalize2mom(act)


def test_identity() -> None:
    act1 = normalize2mom(torch.relu)
    act2 = normalize2mom(act1)

    x = torch.randn(10)
    assert (act1(x) == act2(x)).all()


def test_deterministic() -> None:
    act1 = normalize2mom(torch.tanh)
    act2 = normalize2mom(torch.tanh)

    x = torch.randn(10)
    assert (act1(x) == act2(x)).all()
