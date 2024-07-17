import warnings
from typing import List, Optional, Tuple, Union

from e3nn import o3
from ._irreps_array import IrrepsArray, _standardize_axis
import torch
import numpy as np


def from_chunks(
    irreps: o3.Irreps,
    chunks: List[Optional[torch.Tensor]],
    leading_shape: Tuple[int, ...],
    dtype=torch.get_default_dtype(),
) -> IrrepsArray:
    r"""Create an IrrepsArray from a list of arrays.

    Args:
        irreps (Irreps): irreps
        chunks (list of optional `jax.Array`): list of arrays
        leading_shape (tuple of int): leading shape of the arrays (without the irreps)

    Returns:
        IrrepsArray
    """
    irreps = o3.Irreps(irreps)
    if len(irreps) != len(chunks):
        raise ValueError(f"e3nn.from_chunks: len(irreps) != len(chunks), {len(irreps)} != {len(chunks)}")

    if not all(x is None or hasattr(x, "shape") for x in chunks):
        raise ValueError(f"e3nn.from_chunks: chunks contains non-array elements type={[type(x) for x in chunks]}")

    if not all(x is None or x.shape == leading_shape + (mul, ir.dim) for x, (mul, ir) in zip(chunks, irreps)):
        raise ValueError(
            f"e3nn.from_chunks: chunks shapes {[None if x is None else x.shape for x in chunks]} "
            f"incompatible with leading shape {leading_shape} and irreps {irreps}. "
            f"Expecting {[leading_shape + (mul, ir.dim) for (mul, ir) in irreps]}."
        )

    for x in chunks:
        if x is not None:
            dtype = x.dtype
            break

    if dtype is None:
        raise ValueError("e3nn.from_chunks: Need to specify dtype if chunks is empty or contains only None.")

    if irreps.dim > 0:
        array = torch.cat(
            [
                (
                    torch.zeros(leading_shape + (mul_ir.dim,), dtype=dtype)  # TODO: Add device
                    if x is None
                    else x.reshape(leading_shape + (mul_ir.dim,))
                )
                for mul_ir, x in zip(irreps, chunks)
            ],
            dim=-1,
        )
    else:
        array = torch.zeros(leading_shape + (0,), dtype=dtype)  # TODO: Add device

    zero_flags = tuple(x is None for x in chunks)

    return IrrepsArray(irreps, array, zero_flags=zero_flags, chunks=chunks)


def as_irreps_array(array: Union[torch.Tensor, IrrepsArray]):
    """Convert an array to an IrrepsArray.

    Args:
        array (jax.Array or IrrepsArray): array to convert

    Returns:
        IrrepsArray
    """
    if isinstance(array, IrrepsArray):
        return array

    array = torch.Tensor(array)

    if array.ndim == 0:
        raise ValueError("e3nn.as_irreps_array: Cannot convert an array of rank 0 to an IrrepsArray.")

    return IrrepsArray(f"{array.shape[-1]}x0e", array)


def zeros(irreps: o3.Irreps, leading_shape: Tuple = (), dtype=torch.get_default_dtype()) -> IrrepsArray:
    r"""Create an IrrepsArray of zeros."""
    irreps = o3.Irreps(irreps)
    array = torch.zeros(leading_shape + (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array, zero_flags=(True,) * len(irreps))


def zeros_like(irreps_array: IrrepsArray) -> IrrepsArray:
    r"""Create an IrrepsArray of zeros with the same shape as another IrrepsArray."""
    return zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype)


def ones(
    irreps: o3.Irreps,
    leading_shape: Tuple = (),
    dtype=torch.get_default_dtype(),
) -> IrrepsArray:
    r"""Create an IrrepsArray of ones."""
    irreps = o3.Irreps(irreps)
    array = torch.ones(leading_shape + (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array, zero_flags=(False,) * len(irreps))


def ones_like(irreps_array: IrrepsArray) -> IrrepsArray:
    r"""Create an IrrepsArray of ones with the same shape as another IrrepsArray."""
    return ones(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype)


def _align_two_irreps(irreps1: o3.Irreps, irreps2: o3.Irreps) -> Tuple[o3.Irreps, o3.Irreps]:
    assert irreps1.num_irreps == irreps2.num_irreps

    irreps1 = list(irreps1)
    irreps2 = list(irreps2)

    i = 0
    while i < min(len(irreps1), len(irreps2)):
        mul_1, ir_1 = irreps1[i]
        mul_2, ir_2 = irreps2[i]

        if mul_1 < mul_2:
            irreps2[i] = (mul_1, ir_2)
            irreps2.insert(i + 1, (mul_2 - mul_1, ir_2))

        if mul_2 < mul_1:
            irreps1[i] = (mul_2, ir_1)
            irreps1.insert(i + 1, (mul_1 - mul_2, ir_1))

        i += 1

    assert [mul for mul, _ in irreps1] == [mul for mul, _ in irreps2]
    return o3.Irreps(irreps1), o3.Irreps(irreps2)


def _align_two_irreps_arrays(input1: IrrepsArray, input2: IrrepsArray) -> Tuple[IrrepsArray, IrrepsArray]:
    irreps1, irreps2 = _align_two_irreps(input1.irreps, input2.irreps)
    input1 = input1.rechunk(irreps1)
    input2 = input2.rechunk(irreps2)

    assert [mul for mul, _ in input1.irreps] == [mul for mul, _ in input2.irreps]
    return input1, input2


def _reduce(
    op,
    array: IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> IrrepsArray:
    axis = _standardize_axis(axis, array.ndim)

    if axis == ():
        return array

    if axis[-1] < array.ndim - 1:
        # irrep dimension is not affected by mean
        return IrrepsArray(
            array.irreps,
            op(array.array, axis=axis, keepdims=keepdims),
            zero_flags=array.zero_flags,
            chunks=[None if x is None else op(x, axis=axis, keepdims=keepdims) for x in array.chunks],
        )

    array = _reduce(op, array, axis=axis[:-1], keepdims=keepdims)
    return from_chunks(
        o3.Irreps([(1, ir) for _, ir in array.irreps]),
        [None if x is None else op(x, axis=-2, keepdims=True) for x in array.chunks],
        array.shape[:-1],
        array.dtype,
    )


def mean(
    array: IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> IrrepsArray:
    """Mean of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the mean is computed.

    Returns:
        `IrrepsArray`: mean of the input array
    """
    return _reduce(torch.mean, array, axis, keepdims)


def sum(
    array: IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> IrrepsArray:
    """Sum of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the sum is computed.

    Returns:
        `IrrepsArray`: sum of the input array
    """
    return _reduce(torch.sum, array, axis, keepdims)


def norm(array: IrrepsArray, *, squared: bool = False, per_irrep: bool = True) -> IrrepsArray:
    """Norm of IrrepsArray.

    Args:
        array (IrrepsArray): input array
        squared (bool): if True, return the squared norm
        per_irrep (bool): if True, return the norm of each irrep individually

    Returns:
        IrrepsArray: norm of the input array
    """

    def f(x):
        if x is None:
            return None

        x = torch.sum(torch.conj(x) * x, axis=-1, keepdims=True)
        if not squared:
            x_safe = torch.where(x == 0.0, 1.0, x)
            x_safe = torch.sqrt(x_safe)
            x = torch.where(x == 0.0, 0.0, x_safe)
        return x

    if per_irrep:
        return o3.from_chunks(
            [(mul, "0e") for mul, _ in array.irreps],
            [f(x) for x in array.chunks],
            array.shape[:-1],
            array.dtype,
        )
    else:
        return IrrepsArray("0e", f(array.array))


def dot(a: IrrepsArray, b: IrrepsArray, per_irrep: bool = False) -> IrrepsArray:
    """Dot product of two IrrepsArray.

    Args:
        a (IrrepsArray): first array (this array get complex conjugated)
        b (IrrepsArray): second array
        per_irrep (bool): if True, return the dot product of each irrep individually

    Returns:
        IrrepsArray: dot product of the two input arrays, as a scalar
    """

    a = a.simplify()
    b = b.simplify()

    if a.irreps != b.irreps:
        raise ValueError("Dot product is only defined for IrrepsArray with the same irreps.")

    if per_irrep:
        out = []
        dtype = a.dtype
        for x, y in zip(a.chunks, b.chunks):
            if x is None or y is None:
                out.append(None)
            else:
                out.append(torch.sum(torch.conj(x) * y, axis=-1, keepdims=True))
                dtype = out[-1].dtype
        return from_chunks(
            [(mul, "0e") for mul, _ in a.irreps],
            out,
            a.shape[:-1],
            dtype,
        )
    else:
        out = 0.0
        for x, y in zip(a.chunks, b.chunks):
            if x is None or y is None:
                continue
            out = out + torch.sum(torch.conj(x) * y, axis=(-2, -1))
        if isinstance(out, float):
            shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
            return zeros("0e", shape, dtype=a.dtype)
        return IrrepsArray("0e", out[..., None])


def normal(
    irreps: o3.Irreps,
    leading_shape: Tuple[int, ...] = (),
    *,
    normalize: bool = False,
    normalization: Optional[str] = "component",
    dtype: Optional[torch.dtype] = None,
) -> IrrepsArray:
    r"""Random array with normal distribution.

    Args:
        irreps (Irreps): irreps of the output array
        leading_shape (tuple of int): shape of the leading dimensions
        normalize (bool): if True, normalize the output array
        normalization (str): normalization of the output array, ``"component"`` or ``"norm"``
            This parameter is ignored if ``normalize=False``.
            This parameter only affects the variance distribution.

    Returns:
        IrrepsArray: random array
    """
    irreps = o3.Irreps(irreps)
    if normalize:
        list = []
        for mul, ir in irreps:
            r = torch.randn(leading_shape + (mul, ir.dim), dtype=dtype)
            r = r / torch.linalg.norm(r, axis=-1, keepdims=True)
            list.append(r)
        return o3.experimental.from_chunks(irreps, list, leading_shape, dtype)
    else:
        if normalization == "component":
            return IrrepsArray(
                irreps,
                torch.randn(leading_shape + (irreps.dim,), dtype=dtype),
            )
        elif normalization == "norm":
            list = []
            for mul, ir in irreps:
                r = torch.randn(leading_shape + (mul, ir.dim), dtype=dtype)
                r = r / np.sqrt(ir.dim)
                list.append(r)
            return o3.experimental.from_chunks(irreps, list, leading_shape, dtype)
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")
