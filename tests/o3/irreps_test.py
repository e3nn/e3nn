import pytest

from e3nn import o3


def test_creation():
    o3.Irrep(3, 1)
    ir = o3.Irrep("3e")
    o3.Irrep(ir)
    assert o3.Irrep('10o') == o3.Irrep(10, -1)
    assert o3.Irrep("1y") == o3.Irrep("1o")

    irreps = o3.Irreps(ir)
    o3.Irreps(irreps)
    o3.Irreps([(32, (4, -1))])
    o3.Irreps("11e")
    assert o3.Irreps("16x1e + 32 x 2o") == o3.Irreps([(16, 1, 1), (32, 2, -1)])
    o3.Irreps(["1e", '2o'])
    o3.Irreps([(16, "3e"), '1e'])
    o3.Irreps([(16, "3e"), '1e', (256, 1, -1)])


def test_slice():
    irreps = o3.Irreps("16x1e + 3e + 2e + 5o")
    assert isinstance(irreps[2:], o3.Irreps)


def test_cat():
    irreps = o3.Irreps("4x1e + 6x2e + 12x2o") + o3.Irreps("1x1e + 2x2e + 12x2o")
    assert len(irreps) == 6
    assert irreps.num_irreps == 4 + 6 + 12 + 1 + 2 + 12


def test_contains():
    assert o3.Irrep("2e") in o3.Irreps("3x0e + 2x2e + 1x3o")
    assert o3.Irrep("2o") not in o3.Irreps("3x0e + 2x2e + 1x3o")


@pytest.mark.xfail()
def test_fail1():
    o3.Irreps([(32, 1)])
