import warnings
from typing import List, Optional, Tuple, Union

from e3nn import o3
import torch

def from_chunks(
    irreps: o3.Irreps,
    chunks: List[Optional[torch.Tensor]],
    leading_shape: Tuple[int, ...],
    dtype=torch.get_default_dtype(),
) -> o3.IrrepsArray:
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
        raise ValueError(
            f"e3nn.from_chunks: len(irreps) != len(chunks), {len(irreps)} != {len(chunks)}"
        )

    if not all(x is None or hasattr(x, "shape") for x in chunks):
        raise ValueError(
            f"e3nn.from_chunks: chunks contains non-array elements type={[type(x) for x in chunks]}"
        )

    if not all(
        x is None or x.shape == leading_shape + (mul, ir.dim)
        for x, (mul, ir) in zip(chunks, irreps)
    ):
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
        raise ValueError(
            "e3nn.from_chunks: Need to specify dtype if chunks is empty or contains only None."
        )

    if irreps.dim > 0:
        array = torch.cat(
            [
                (
                    torch.zeros(leading_shape + (mul_ir.dim,), dtype=dtype) #TODO: Add device
                    if x is None
                    else x.reshape(leading_shape + (mul_ir.dim,))
                )
                for mul_ir, x in zip(irreps, chunks)
            ],
            dim=-1,
        )
    else:
        array = torch.zeros(leading_shape + (0,), dtype=dtype) #TODO: Add device

    zero_flags = tuple(x is None for x in chunks)

    return o3.IrrepsArray(irreps, array, zero_flags=zero_flags, chunks=chunks)


def zeros(
    irreps: o3.Irreps, leading_shape: Tuple = (), dtype = torch.get_default_dtype()
) -> o3.IrrepsArray:
    r"""Create an IrrepsArray of zeros."""
    irreps = o3.Irreps(irreps)
    array = torch.zeros(leading_shape + (irreps.dim,), dtype=dtype)
    return  o3.IrrepsArray(irreps, array, zero_flags=(True,) * len(irreps))


def zeros_like(irreps_array: o3.IrrepsArray) -> o3.IrrepsArray:
    r"""Create an IrrepsArray of zeros with the same shape as another IrrepsArray."""
    return zeros(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype)

def ones(
    irreps: o3.Irreps, leading_shape: Tuple = (), dtype = torch.get_default_dtype(),
) -> o3.IrrepsArray:
    r"""Create an IrrepsArray of ones."""
    irreps = o3.Irreps(irreps)
    array = torch.ones(leading_shape + (irreps.dim,), dtype=dtype)
    return o3.IrrepsArray(irreps, array, zero_flags=(False,) * len(irreps))


def ones_like(irreps_array: o3.IrrepsArray) -> o3.IrrepsArray:
    r"""Create an IrrepsArray of ones with the same shape as another IrrepsArray."""
    return ones(irreps_array.irreps, irreps_array.shape[:-1], irreps_array.dtype)

def _reduce(
    op,
    array: o3.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> o3.IrrepsArray:
    axis = o3._standardize_axis(axis, array.ndim)

    if axis == ():
        return array

    if axis[-1] < array.ndim - 1:
        # irrep dimension is not affected by mean
        return o3.IrrepsArray(
            array.irreps,
            op(array.array, axis=axis, keepdims=keepdims),
            zero_flags=array.zero_flags,
            chunks=[
                None if x is None else op(x, axis=axis, keepdims=keepdims)
                for x in array.chunks
            ],
        )

    array = _reduce(op, array, axis=axis[:-1], keepdims=keepdims)
    return o3.from_chunks(
        o3.Irreps([(1, ir) for _, ir in array.irreps]),
        [None if x is None else op(x, axis=-2, keepdims=True) for x in array.chunks],
        array.shape[:-1],
        array.dtype,
    )


def mean(
    array: o3.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> o3.IrrepsArray:
    """Mean of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the mean is computed.

    Returns:
        `IrrepsArray`: mean of the input array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0e", jnp.arange(2 * 5).reshape(2, 5))
        >>> e3nn.mean(x, axis=0)
        3x0e+2x0e [2.5 3.5 4.5 5.5 6.5]
        >>> e3nn.mean(x, axis=1)
        1x0e+1x0e
        [[1.  3.5]
         [6.  8.5]]
        >>> e3nn.mean(x)
        1x0e+1x0e [3.5 6. ]
    """
    return _reduce(torch.mean, array, axis, keepdims)


def sum(
    array: o3.IrrepsArray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> o3.IrrepsArray:
    """Sum of IrrepsArray along the specified axis.

    Args:
        array (`IrrepsArray`): input array
        axis (optional int or tuple of ints): axis along which the sum is computed.

    Returns:
        `IrrepsArray`: sum of the input array

    Examples:
        >>> x = e3nn.IrrepsArray("3x0e + 2x0e", jnp.arange(2 * 5).reshape(2, 5))
        >>> e3nn.sum(x, axis=0)
        3x0e+2x0e [ 5  7  9 11 13]
        >>> e3nn.sum(x, axis=1)
        1x0e+1x0e
        [[ 3  7]
         [18 17]]
        >>> e3nn.sum(x)
        1x0e+1x0e [21 24]
        >>> e3nn.sum(x.regroup())
        1x0e [45]
    """
    return _reduce(torch.sum, array, axis, keepdims)

def norm(
        array: o3.IrrepsArray, *, squared: bool = False, per_irrep: bool = True
        ) -> o3.IrrepsArray:
            """Norm of IrrepsArray.

            Args:
                array (IrrepsArray): input array
                squared (bool): if True, return the squared norm
                per_irrep (bool): if True, return the norm of each irrep individually

            Returns:
                IrrepsArray: norm of the input array

            Examples:
                >>> x = e3nn.IrrepsArray("2x0e + 1e + 2e", jnp.arange(10.0))
                >>> e3nn.norm(x)
                2x0e+1x0e+1x0e [ 0.         1.         5.3851647 15.9687195]

                >>> e3nn.norm(x, squared=True)
                2x0e+1x0e+1x0e [  0.   1.  29. 255.]

                >>> e3nn.norm(x, per_irrep=False)
                1x0e [16.881943]
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
                return o3.IrrepsArray("0e", f(array.array))