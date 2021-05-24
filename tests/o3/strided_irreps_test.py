import pytest

from e3nn import o3


def test_creation():
    o3.StridedIrreps(o3.Irreps("3x0e + 3x0o + 3x4e"))
    o3.StridedIrreps("3x0e + 3x0o + 3x4o")
    o3.StridedIrreps([(4, (2, 1)), (4, (0, 1)), (4, (2, -1))])
    with pytest.raises(ValueError):
        o3.StridedIrreps(o3.Irreps("3x0e + 13x0o + 3x4e"))
    with pytest.raises(ValueError):
        o3.StridedIrreps("3x0e + 3x0o + 34x4o")
    with pytest.raises(ValueError):
        o3.StridedIrreps([(4, (2, 1)), (4, (0, 1)), (49, (2, -1))])


def test_properties():
    irreps = o3.Irreps("3x0e + 3x0o + 3x4e")
    strided = o3.StridedIrreps(irreps)
    assert strided.dim == irreps.dim
    assert strided.base_irreps == o3.Irreps("0e + 0o + 4e")
    assert strided.mul == 3
    assert strided.ls == [0]*6 + [4]*3


def test_empty_irreps():
    assert o3.StridedIrreps() == o3.StridedIrreps("") == o3.StridedIrreps([])
    assert len(o3.StridedIrreps()) == 0
    assert o3.StridedIrreps().dim == 0
    assert o3.StridedIrreps().ls == []
    assert o3.StridedIrreps().num_irreps == 0
    assert o3.StridedIrreps().mul == 0


def test_zero_muls():
    s1 = o3.StridedIrreps([(4, (2, 1)), (0, (0, 1)), (4, (2, -1))])
    s2 = o3.StridedIrreps([(4, (2, 1)), (4, (2, -1))])
    assert s1.dim == s2.dim
    assert s1.mul == s2.mul == 4
    assert s1.num_irreps == s2.num_irreps == 8


def test_arithmetic():
    a = o3.StridedIrreps("3x0e + 3x0o + 3x4o")
    assert a * 3 == o3.StridedIrreps("9x0e + 9x0o + 9x4o")
    assert 4 * a == o3.StridedIrreps("12x0e + 12x0o + 12x4o")
    assert a + o3.StridedIrreps("3x3o") == o3.StridedIrreps("3x0e + 3x0o + 3x4o + 3x3o")
    with pytest.raises(ValueError):
        a + o3.StridedIrreps("7x3o")
