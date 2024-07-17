from math import prod

import torch
import numpy as np
import pytest

from e3nn import o3
from torch.utils._pytree import tree_map


def _shape(x):
    return None if x is None else getattr(x, "shape")


def test_empty():
    x = o3.experimental.from_chunks("", [], (2, 2), torch.float32)
    assert x.irreps == o3.Irreps([])
    assert x.shape == (2, 2, 0)


def test_convert():
    id = o3.experimental.from_chunks("10x0e + 10x0e", [None, torch.ones((1, 10, 1))], (1,))
    assert tree_map(_shape, id.rechunk("0x0e + 20x0e + 0x0e").chunks) == [None, (1, 20, 1), None]
    assert tree_map(_shape, id.rechunk("7x0e + 4x0e + 9x0e").chunks) == [None, (1, 4, 1), (1, 9, 1)]

    id = o3.from_chunks("10x0e + 10x1e", [None, torch.ones((1, 10, 3))], (1,))
    assert tree_map(_shape, id.rechunk("5x0e + 5x0e + 5x1e + 5x1e").chunks) == [
        None,
        None,
        (1, 5, 3),
        (1, 5, 3),
    ]

    id = o3.experimental.zeros("10x0e + 10x1e", ())
    id = id.rechunk("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")

    a = o3.experimental.from_chunks(
        "            10x0e  +  0x0e +1x1e  +     0x0e    +          9x1e           + 0x0e",
        [
            torch.ones((2, 10, 1)),
            None,
            None,
            torch.ones((2, 0, 1)),
            torch.ones((2, 9, 3)),
            None,
        ],
        (2,),
    )
    b = a.rechunk("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")
    b = o3.experimental.from_chunks(b.irreps, b.chunks, b.shape[:-1])

    torch.testing.assert_allclose(a.array, b.array)


def test_indexing():
    x = o3.experimental.IrrepsArray("2x0e + 1x0e", torch.Tensor([[1.0, 2, 3], [4.0, 5, 6]]))
    assert x.shape == (2, 3)
    torch.testing.assert_allclose(x[0].array, torch.Tensor([1.0, 2, 3]))
    torch.testing.assert_allclose(x[1, "1x0e"].array, torch.Tensor([6.0]))
    torch.testing.assert_allclose(x[:, "1x0e"].array, torch.Tensor([[3.0], [6.0]]))
    torch.testing.assert_allclose(x[..., "1x0e"].array, torch.Tensor([[3.0], [6.0]]))
    torch.testing.assert_allclose(x[..., 1, "1x0e"].array, torch.Tensor([6.0]))
    torch.testing.assert_allclose(x[..., 1, 2:].array, torch.Tensor([6.0]))
    torch.testing.assert_allclose(x[..., 1, 2:3].array, torch.Tensor([6.0]))
    torch.testing.assert_allclose(x[..., 1, -1:50].array, torch.Tensor([6.0]))
    torch.testing.assert_allclose(x[..., 1, "2x0e + 1x0e"].array, torch.Tensor([4, 5, 6.0]))
    torch.testing.assert_allclose(x[..., :1].array, torch.Tensor([[1.0], [4.0]]))
    torch.testing.assert_allclose(x[..., 1:].array, torch.Tensor([[2, 3], [5.0, 6]]))

    x = o3.experimental.IrrepsArray("2x0e + 1x2e", torch.arange(3 * 4 * 7).reshape((3, 4, 7)))
    torch.testing.assert_allclose(x[..., 1, -5:].array, x[:3, 1, "2e"].array)

    x = o3.zeros("2x1e + 2x1e", (3, 3))
    with pytest.raises(IndexError):
        x[..., "2x1e"]

    with pytest.raises(IndexError):
        x[..., :2]

    x = o3.IrrepsArray(
        "2x1e + 2x1e",
        torch.Tensor([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]),
    )
    assert x[3:-3].irreps == o3.Irreps("1e + 1e")  # TODO: e3nn-jax is able to make it work without o3.Irreps
    torch.testing.assert_allclose(x[3:-3].array, torch.Tensor([1.1, 1.2, 1.3, 2.1, 2.2, 2.3]))


def test_indexing2():
    x = o3.experimental.IrrepsArray("2x0e + 1x0e", torch.ones((2, 2, 3)))
    y = x[0]
    assert y.shape == (2, 3)
    assert y.chunks[0].shape == (2, 2, 1)  # TODO: y._chunks works in e3nn-jax


def test_reductions():
    x = o3.IrrepsArray("2x0e + 1x1e", torch.Tensor([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))
    assert o3.experimental.sum(x).irreps == o3.Irreps("0e + 1e")
    torch.testing.assert_allclose(o3.sum(x).array, torch.Tensor([12.0, 9, 10, 11]))
    torch.testing.assert_allclose(o3.experimental.sum(x, axis=0).array, torch.Tensor([5.0, 7, 9, 10, 11]))
    torch.testing.assert_allclose(o3.experimental.sum(x, axis=1).array, torch.Tensor([[3.0, 3, 4, 5], [9.0, 6, 6, 6]]))

    np.testing.assert_allclose(o3.experimental.mean(x, axis=1).array, torch.Tensor([[1.5, 3, 4, 5], [4.5, 6, 6, 6]]))


def test_operators():
    x = o3.experimental.IrrepsArray("2x0e + 1x1e", torch.Tensor([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))
    y = o3.experimental.IrrepsArray("2x0e + 1x1o", torch.Tensor([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))

    with pytest.raises(ValueError):
        x + 1

    # o3.norm(x) + 1 # Need to debug

    assert (x + x).shape == x.shape

    with pytest.raises(ValueError):
        x + y

    assert (x - x).shape == x.shape

    with pytest.raises(ValueError):
        x - y

    assert (x * 2.0).shape == x.shape
    assert (x / 2.0).shape == x.shape
    assert (x * torch.Tensor([[2], [3.0]])).shape == x.shape

    with pytest.raises(ValueError):
        x * torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    x * torch.Tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        1.0 / x

    1.0 / o3.norm(x)

    torch.set_default_dtype(torch.float64)
    torch.testing.assert_allclose(o3.norm(x / o3.norm(x)).array, 1)
    torch.set_default_dtype(torch.float32)


def test_set():
    x = o3.experimental.IrrepsArray("0e + 1e", torch.arange(3 * 4 * 4).reshape((3, 4, 4)))

    x[0, 1] = 0
    assert x.shape == x.shape
    np.testing.assert_allclose(y[0, 1].array, 0)
    np.testing.assert_allclose(y[0, 1].chunks[0], 0)
    np.testing.assert_allclose(y[0, 1].chunks[1], 0)
    np.testing.assert_allclose(y[0, 2].array, x[0, 2].array)
    np.testing.assert_allclose(y[0, 2].chunks[0], x[0, 2].chunks[0])
    np.testing.assert_allclose(y[0, 2].chunks[1], x[0, 2].chunks[1])

    v = o3.experimental.IrrepsArray("0e + 1e", torch.arange(4 * 4).reshape((4, 4)))
    x[1] = v
    assert y.shape == x.shape
    torch.testing.assert_allclose(x[1].array, v.array)
    torch.testing.assert_allclose(x[1].chunks[0], v.chunks[0])
    torch.testing.assert_allclose(x[1].chunks[1], v.chunks[1])
    torch.testing.assert_allclose(x[0].array, x[0].array)
    torch.testing.assert_allclose(x[0].chunks[0], x[0].chunks[0])
    torch.testing.assert_allclose(x[0].chunks[1], x[0].chunks[1])


def test_at_add():
    def f(*shape):
        return 1.0 + torch.arange(prod(shape)).reshape(shape)

    x = o3.experimental.from_chunks(
        "1e + 0e + 0e + 0e",
        [None, None, f(2, 1, 1), f(2, 1, 1)],
        (2,),
    )
    v = o3.experimental.from_chunks("1e + 0e + 0e + 0e", [None, f(1, 1), None, f(1, 1)], ())
    x[0] += v
    y2 = o3.experimental.IrrepsArray(x.irreps, x.array.at[0].add(v.array))
    np.testing.assert_array_equal(y1.array, y2.array)
    assert y1.chunks[0] is None
    assert y1.chunks[1] is not None
    assert y1.chunks[2] is not None
    assert y1.chunks[3] is not None
    np.testing.assert_allclose(0, y2.chunks[0])
    np.testing.assert_array_equal(y1.chunks[1], y2.chunks[1])
    np.testing.assert_array_equal(y1.chunks[2], y2.chunks[2])
    np.testing.assert_array_equal(y1.chunks[3], y2.chunks[3])


def test_slice_by_mul():
    x = o3.experimental.IrrepsArray("3x0e + 4x1e", torch.arange(3 + 4 * 3))
    y = x.slice_by_mul[2:4]
    assert y.irreps == o3.Irreps("0e + 1e")
    torch.testing.assert_allclose(y.array, torch.Tensor([2.0, 3.0, 4.0, 5.0]))

    y = x.slice_by_mul[:0]
    assert y.irreps == o3.Irreps("")
    assert y.array.shape == (0,)
    assert len(y.chunks) == 0


def test_norm():
    x = o3.experimental.IrrepsArray("2x0e + 1x1e", torch.Tensor([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))

    assert o3.experimental.norm(x).shape == (2, 3)
    assert o3.experimental.norm(x, per_irrep=True).shape == (2, 3)
    assert o3.experimental.norm(x, per_irrep=False).shape == (2, 1)

    x = o3.experimental.from_chunks("2x0e + 1x1e", [None, None], (2,), dtype=torch.complex64)

    assert o3.experimental.norm(x).shape == (2, 3)


def test_dot():
    x = o3.experimental.IrrepsArray("2x0e + 1x1e", torch.Tensor([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))
    y = o3.experimental.IrrepsArray(
        "2x0e + 1x1e", torch.tensor([[1.0j, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]], dtype=torch.complex64)
    )

    assert o3.experimental.dot(x, y).shape == (2, 1)
    assert o3.experimental.dot(x, y, per_irrep=True).shape == (2, 3)
    assert o3.experimental.dot(x, y, per_irrep=False).shape == (2, 1)

    y = o3.experimental.from_chunks("2x0e + 1x1e", [None, None], (2,), dtype=torch.complex64)

    assert o3.experimental.dot(x, y).shape == (2, 1)
