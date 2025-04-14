import torch
from e3nn import o3
from e3nn.o3._irreps import _MulIr, Irreps
from typing import Optional, Tuple, Union, List, Any
from torch.utils._pytree import tree_map
import operator
import numpy as np
import warnings


def _is_ellipse(x):
    return type(x) == type(Ellipsis)


def _is_none_slice(x):
    return isinstance(x, slice) and x == slice(None)


class IrrepsArray(object):
    def __init__(
        self,
        irreps: Irreps,
        array: torch.Tensor,
        zero_flags: Optional[Tuple] = None,
        chunks: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        self.array = torch.as_tensor(array)
        self.irreps = Irreps(irreps)
        self._zero_flags = zero_flags
        self._chunks = chunks

    def __repr__(self):  # noqa: D105
        r = str(self.array)
        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a.irreps for a in args if hasattr(a, "irreps"))
        args = [getattr(a, "array", a) for a in args]
        ret = func(*args, **kwargs)
        return IrrepsArray(ret, metadata=metadatas[0])

    def __post_init__(self):
        if hasattr(self.array, "shape"):
            if self.array.shape[-1] != self.irreps.dim:
                raise ValueError(
                    f"IrrepsArray: Array shape {self.array.shape} incompatible with irreps {self.irreps}. "
                    f"{self.array.shape[-1]} != {self.irreps.dim}"
                )
        if self.zero_flags is not None:
            if len(self.zero_flags) != len(self.irreps):
                raise ValueError(f"IrrepsArray: len(zero_flags) != len(irreps), {len(self.zero_flags)} != {len(self.irreps)}")

    @staticmethod
    def from_list(
        irreps: o3.Irreps,
        chunks: List[Optional[torch.Tensor]],
        leading_shape: Tuple[int, ...],
        dtype=None,
        *,
        backend=None,
    ):
        warnings.warn(
            "IrrepsArray.from_list is deprecated, use e3nn.from_chunks instead.",
            DeprecationWarning,
        )
        return o3.experimental.from_chunks(irreps, chunks, leading_shape, dtype, backend=backend)

    @property
    def chunks(self) -> List[Optional[torch.Tensor]]:
        leading_shape = self.array.shape[:-1]
        if self.zero_flags is None:
            zeros = [False] * self.irreps.dim
        else:
            zeros = self.zero_flags

        if self.irreps.dim == 1:
            mul, ir = self.irreps[0]
            if zeros[0]:
                return [None]
            return [torch.reshape(self.array, leading_shape + (mul, ir.dim))]
        else:
            return [
                None if zero else torch.reshape(self.array[..., i], leading_shape + (mul, ir.dim))
                for zero, i, (mul, ir) in zip(zeros, self.irreps.slices(), self.irreps)
            ]

    def simplify(self) -> "IrrepsArray":
        r"""Simplify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0e + 0e", torch.ones(3)).simplify()
            3x0e [1. 1. 1.]

            >>> IrrepsArray("0e + 0x1e + 0e", torch.ones(2)).simplify()
            2x0e [1. 1.]
        """
        return self.rechunk(self.irreps.simplify())

    def sort(self) -> "IrrepsArray":
        r"""Sort the irreps.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", torch.arange(6)).sort()
            1x0e+2x0e+1x1o [0 4 5 1 2 3]
        """
        irreps, p, inv = self.irreps.sort()
        return o3.experimental.from_chunks(irreps, [self.chunks[i] for i in inv], self.shape[:-1], self.dtype)

    def regroup(self) -> "IrrepsArray":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sorted` followed by :meth:`simplify`.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", torch.arange(6)).regroup()
            3x0e+1x1o [0 4 5 1 2 3]
        """
        return self.sort().simplify()

    def at(self):
        return _IndexUpdateHelper(self)

    def to(self, dtype) -> "IrrepsArray":
        r"""Change the dtype of the array.

        Args:
            dtype (dtype): new dtype

        Returns:
            IrrepsArray: new IrrepsArray
        """
        return IrrepsArray(
            irreps=self.irreps,
            array=self.array.to(dtype),
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: x if x is None else x.to(dtype), self._chunks),
        )

    def sorted(self) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.sorted is deprecated, use IrrepsArray.sort instead.",
            DeprecationWarning,
        )
        return self.sort()

    @property
    def slice_by_mul(self):
        r"""Return the slice with respect to the multiplicities.

        See also:
            :meth:`Irreps.slice_by_mul`
        """
        return _MulIndexSliceHelper(self)

    @property
    def slice_by_dim(self):
        r"""Same as ``__getitem__`` in the irreps dimension.

        See also:
            :meth:`Irreps.slice_by_dim`
        """
        return _DimIndexSliceHelper(self)

    @property
    def slice_by_chunk(self):
        r"""Return the slice with respect to the chunks.

        See also:
            :meth:`Irreps.slice_by_chunk`
        """
        return _ChunkIndexSliceHelper(self)

    @property
    def zero_flags(self):
        if self._zero_flags is None:
            return (False,) * self.irreps.dim
        return self._zero_flags

    @property
    def shape(self):
        r"""Shape. Equivalent to ``self.array.shape``."""
        return self.array.shape

    @property
    def dtype(self):
        r"""dtype. Equivalent to ``self.array.dtype``."""
        return self.array.dtype

    @property
    def ndim(self):
        r"""Number of dimensions. Equivalent to ``self.array.ndim``."""
        return len(self.shape)

    def __len__(self):  # noqa: D105
        return len(self.array)

    def __eq__(self: "IrrepsArray", other: Union["IrrepsArray", torch.Tensor]) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, IrrepsArray):
            if self.irreps != other.irreps:
                raise ValueError(
                    "IrrepsArray({self.irreps}, shape={self.shape}) == IrrepsArray({other.irreps}) is not equivariant."
                )

            leading_shape = torch.broadcast_shapes(self.shape[:-1], other.shape[:-1])

            def eq(mul: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if x is None and y is None:
                    return torch.ones(leading_shape + (mul,), dtype=bool)
                if x is None:
                    x = 0.0
                if y is None:
                    y = 0.0

                return torch.all(x == y, axis=-1)

            chunks = [eq(mul, x, y)[..., None] for (mul, ir), x, y in zip(self.irreps, self.chunks, other.chunks)]
            return o3.experimental.from_chunks([(mul, "0e") for mul, _ in self.irreps], chunks, leading_shape, bool)

        other = torch.from_numpy(np.asarray(other))
        if self.irreps.lmax > 0 or (other.ndim > 0 and other.shape[-1] != 1):
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) == scalar(shape={other.shape}) is not equivariant."
            )
        return IrrepsArray(self.irreps, self.array == other)

    def __neg__(self: "IrrepsArray") -> "IrrepsArray":
        return IrrepsArray(
            self.irreps,
            -self.array,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: -x, self._chunks),
        )

    def __add__(self: "IrrepsArray", other: Union["IrrepsArray", torch.Tensor, float, int]) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = torch.from_numpy(np.asarray(other))
                return IrrepsArray(self.irreps, self.array + other)
            raise ValueError(f"IrrepsArray({self.irreps}, shape={self.shape}) + scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) + IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        chunks = None
        if self._chunks is not None and other._chunks is not None:
            chunks = [y if x is None else x if y is None else x + y for x, y in zip(self._chunks, other._chunks)]

        return IrrepsArray(self.irreps, self.array + other.array, zero_flags=zero_flags, chunks=chunks)

    def __radd__(self: "IrrepsArray", other: Union[torch.Tensor, float, int]) -> "IrrepsArray":
        return self + other

    def __sub__(self: "IrrepsArray", other: Union["IrrepsArray", torch.Tensor, float, int]) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = torch.from_numpy(np.asarray(other))
                return IrrepsArray(irreps=self.irreps, array=self.array - other)
            raise ValueError(f"IrrepsArray({self.irreps}, shape={self.shape}) - scalar is not equivariant.")

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) - IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        chunks = None
        if self.chunks is not None and other.chunks is not None:
            chunks = [x if y is None else -y if x is None else x - y for x, y in zip(self.chunks, other.chunks)]

        return IrrepsArray(self.irreps, self.array - other.array, zero_flags=zero_flags, chunks=chunks)

    def __rsub__(self: "IrrepsArray", other: Union[torch.Tensor, float, int]) -> "IrrepsArray":
        return -self + other

    def __mul__(self: "IrrepsArray", other: Union["IrrepsArray", torch.Tensor]) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, IrrepsArray):
            if self.irreps.num_irreps != other.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) * IrrepsArray({other.irreps}) "
                    "only works if the number of irreps is the same."
                )
            irreps_out = o3.experimental.elementwise_tensor_product(self.irreps, other.irreps)
            if irreps_out.num_irreps != self.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) * IrrepsArray({other.irreps}) "
                    "is only supported for scalar * irreps and irreps * scalar. "
                    "To perform irreps * irreps use e3nn.elementwise_tensor_product or e3nn.tensor_product."
                )
            return o3.experimental.elementwise_tensor_product(self, other)

        other = torch.from_numpy(np.asarray(other))
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", other)
            return o3.experimental.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) * scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(
            self.irreps,
            self.array * other,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: None if x is None else x * other[..., None], self._chunks),
        )

    def __rmul__(self: "IrrepsArray", other: torch.Tensor) -> "IrrepsArray":  # noqa: D105
        return self * other

    def __truediv__(self: "IrrepsArray", other: Union["IrrepsArray", torch.Tensor]) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, IrrepsArray):
            if len(other.irreps) == 0 or other.irreps.lmax > 0 or self.irreps.num_irreps != other.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) / IrrepsArray({other.irreps}) is not equivariant."
                )

            if any(x is None for x in other.chunks):
                raise ValueError("There are deterministic Zeros in the array of the lhs. Cannot divide by Zero.")
            other = 1.0 / other
            return o3.experimental.elementwise_tensor_product(self, other)

        other = torch.from_numpy(np.asarray(other))
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", 1.0 / other)
            return o3.experimental.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) / scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(
            self.irreps,
            self.array / other,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: None if x is None else x / other[..., None], self._chunks),
        )

    def __rtruediv__(self: "IrrepsArray", other: torch.Tensor) -> "IrrepsArray":  # noqa: D105
        other = torch.from_numpy(np.asarray(other))
        if self.irreps.lmax > 0:
            raise ValueError(
                f"scalar(shape={other.shape}) / IrrepsArray({self.irreps}, shape={self.shape}) is not equivariant."
            )
        if any(x is None for x in self.chunks):
            raise ValueError("There are deterministic Zeros in the array of the lhs. Cannot divide by Zero.")

        return IrrepsArray(self.irreps, other / self.array)

    def rechunk(self, irreps: Irreps) -> "IrrepsArray":
        r"""Rechunk the array with new (equivalent) irreps.

        Args:
            irreps (Irreps): new irreps

        Returns:
            `IrrepsArray`: new IrrepsArray

        Examples:
            >>> x = e3nn.from_chunks("6x0e + 4x0e", [None, torch.ones((4, 1))], ())
            >>> x.rechunk("5x0e + 5x0e").chunks
            [None, Array([[0.],
                   [1.],
                   [1.],
                   [1.],
                   [1.]], dtype=float32)]
        """
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)

        if self.irreps == irreps:
            return self

        if len(self.irreps) == 0:
            zero_flags = torch.empty((0,), dtype=bool)
        else:
            zero_flags = torch.cat(
                [z * torch.ones(mul * ir.dim, dtype=bool) for z, (mul, ir) in zip(self.zero_flags, self.irreps)]
            )
        zero_flags = [bool(torch.all(zero_flags[s])) for s in irreps.slices()]

        new_chunks = None
        if self._chunks is not None:
            leading_shape = self.shape[:-1]

            new_chunks = []
            current_array = 0

            while len(new_chunks) < len(irreps) and irreps[len(new_chunks)].mul == 0:
                new_chunks.append(None)

            for mul_ir, y in zip(self.irreps, self.chunks):
                mul, _ = mul_ir

                while mul > 0:
                    if isinstance(current_array, int):
                        current_mul = current_array
                    else:
                        current_mul = current_array.shape[-2]

                    needed_mul = irreps[len(new_chunks)].mul - current_mul

                    if mul <= needed_mul:
                        x = y
                        m = mul
                        mul = 0
                    elif mul > needed_mul:
                        if y is None:
                            x = None
                        else:
                            x, y = torch.tensor_split(y, [needed_mul], dim=-2)
                        m = needed_mul
                        mul -= needed_mul

                    if x is None:
                        if isinstance(current_array, int):
                            current_array += m
                        else:
                            current_array = torch.cat(
                                [
                                    current_array,
                                    torch.zeros(leading_shape + (m, mul_ir.ir.dim), dtype=self.dtype),
                                ],
                                dim=-2,
                            )
                    else:
                        if isinstance(current_array, int):
                            if current_array == 0:
                                current_array = x
                            else:
                                current_array = torch.cat(
                                    [
                                        torch.zeros(
                                            leading_shape + (current_array, mul_ir.ir.dim),
                                            dtype=self.dtype,
                                        ),
                                        x,
                                    ],
                                    dim=-2,
                                )
                        else:
                            current_array = torch.cat([current_array, x], dim=-2)

                    if isinstance(current_array, int):
                        if current_array == irreps[len(new_chunks)].mul:
                            new_chunks.append(None)
                            current_array = 0
                    else:
                        if current_array.shape[-2] == irreps[len(new_chunks)].mul:
                            new_chunks.append(current_array)
                            current_array = 0

                    while len(new_chunks) < len(irreps) and irreps[len(new_chunks)].mul == 0:
                        new_chunks.append(None)

            assert current_array == 0

            assert len(new_chunks) == len(irreps)
            for (mul, ir), x, z in zip(irreps, new_chunks, zero_flags):
                if z:
                    assert x is None
                else:
                    assert x.shape[-2:] == (mul, ir.dim)

        return IrrepsArray(irreps, self.array, zero_flags=zero_flags, chunks=new_chunks)

    def broadcast_to(self, shape) -> "IrrepsArray":
        """Broadcast the array to a new shape."""

        assert isinstance(shape, tuple)
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        leading_shape = shape[:-1]
        array = torch.broadcast_to(self.array, leading_shape + (self.irreps.dim,))
        chunks = [None if x is None else torch.broadcast_to(x, leading_shape + x.shape[-2:]) for x in self.chunks]
        return IrrepsArray(self.irreps, array, zero_flags=self.zero_flags, chunks=chunks)

    def __getitem__(self, index) -> "IrrepsArray":  # noqa: D105
        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (o3.Irrep, _MulIr, Irreps, str)):
            if not (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim):
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Irreps index must be the last index, try x[..., {index[-1]}]."
                )

            irreps = Irreps(index[-1])

            ii = [i for i in range(len(self.irreps)) if self.irreps[i: i + len(irreps)] == irreps]
            if len(ii) != 1:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Can't slice with {irreps} "
                    f"because it doesn't appear exactly once in {self.irreps}."
                )
            i = ii[0]

            return IrrepsArray(
                irreps,
                self.array[..., self.irreps[:i].dim : self.irreps[: i + len(irreps)].dim],
                zero_flags=self.zero_flags[i: i + len(irreps)],
                chunks=self.chunks[i: i + len(irreps)],
            )[index[:-1] + (slice(None),)]

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and index[-1].step is None
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            start, stop, _ = index[-1].indices(self.shape[-1])

            irreps_start = None
            irreps_stop = None

            for i in range(len(self.irreps) + 1):
                if self.irreps[:i].dim == start:
                    irreps_start = i

                if irreps_start is None and start < self.irreps[:i].dim:
                    # "2x1e"[3:]
                    mul, ir = self.irreps[i - 1]
                    if (start - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (start - self.irreps[: i - 1].dim) // ir.dim
                        return self.rechunk(self.irreps[: i - 1] + Irreps([(mul1, ir), (mul - mul1, ir)]) + self.irreps[i:])[
                            index
                        ]

                if self.irreps[:i].dim == stop:
                    irreps_stop = i
                    break

                if irreps_stop is None and stop < self.irreps[:i].dim:
                    # "2x1e"[:3]
                    mul, ir = self.irreps[i - 1]
                    if (stop - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (stop - self.irreps[: i - 1].dim) // ir.dim
                        return self.rechunk(self.irreps[: i - 1] + Irreps([(mul1, ir), (mul - mul1, ir)]) + self.irreps[i:])[
                            index
                        ]

            if irreps_start is None or irreps_stop is None:
                raise IndexError(f"Error in IrrepsArray.__getitem__, unable to slice {self.irreps} with {start}:{stop}.")

            return IrrepsArray(
                self.irreps[irreps_start:irreps_stop],
                self.array[..., start:stop],
                zero_flags=self.zero_flags[irreps_start:irreps_stop],
                chunks=self.chunks[irreps_start:irreps_stop],
            )[index[:-1] + (slice(None),)]

        # Prevent None at last index  x[..., None] and x[:, :, None]
        if (len(index[:-1]) == self.ndim or any(map(_is_ellipse, index[:-1]))) and index[-1] is None:
            raise IndexError("Error in IrrepsArray.__getitem__, cannot add a new dimension at the end.")

        # Prevent indexing the last axis
        if (len(index) == self.ndim or any(map(_is_ellipse, index[:-1]))) and not (
            _is_ellipse(index[-1]) or _is_none_slice(index[-1]) or index[-1] is None
        ):
            if isinstance(index[-1], int):
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, integer index in the irreps dimension is not supported, "
                    f"try x[..., {index[-1]}:{index[-1] + 1}] instead."
                )
            raise IndexError(
                f"Error in IrrepsArray.__getitem__, indexing the irreps dimension with [..., {index[-1]}] " "is not supported."
            )

        # Support of x[index, :]
        return IrrepsArray(
            self.irreps,
            self.array[index],
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: None if x is None else x[index + (slice(None),)], self._chunks),
        )

# We purposefully do not register zero_flags
torch.utils._pytree.register_pytree_node(
    IrrepsArray,
    lambda x: ((x.array,), x.irreps),
    lambda irreps, data: IrrepsArray(irreps, data[0]),
)

def _standardize_axis(axis: Union[None, int, Tuple[int, ...]], result_ndim: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(result_ndim))
    try:
        axis = (operator.index(axis),)
    except TypeError:
        axis = tuple(operator.index(i) for i in axis)

    if not all(-result_ndim <= i < result_ndim for i in axis):
        raise ValueError("axis out of range")
    axis = tuple(i % result_ndim for i in axis)

    return tuple(sorted(set(axis)))


class _IndexUpdateHelper:
    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.irreps_array, index)


class _IndexUpdateRef:
    def __init__(self, irreps_array, index) -> None:
        self.irreps_array = irreps_array
        self.index = index

    def set(self, values: Any) -> IrrepsArray:
        index = self.index
        self = self.irreps_array

        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (o3.Irrep, _MulIr, Irreps, str)):
            raise NotImplementedError('x[..., "1e + 2e"] is not implemented')

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and index[-1].step is None
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            raise NotImplementedError("x.at[..., 3:32] is not implemented")

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                raise IndexError(f"Indexing with {index[-1]} in the irreps dimension is not supported.")

        # Support of x.at[index, :].set(0)
        if isinstance(values, (int, float)) and values == 0:
            return IrrepsArray(
                self.irreps,
                array=self.array.at[index].set(0),
                zero_flags=self.zero_flags,
            )

        # Support of x.at[index, :].set(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError("The irreps of the array and the values to set must be the same.")

            values = values.rechunk(self.irreps)

            zero_flags = tuple(x and y for x, y in zip(self.zero_flags, values.zero_flags))
            return IrrepsArray(
                self.irreps,
                self.array.at[index].set(values.array),
                zero_flags=zero_flags,
            )

        raise NotImplementedError(f"x.at[i].set(v) with v={type(values)} is not implemented.")

    def add(self, values: Any) -> IrrepsArray:
        index = self.index
        self = self.irreps_array

        if not isinstance(index, tuple):
            index = (index,)

        # Support of x[..., "1e + 2e"]
        if isinstance(index[-1], (o3.Irrep, _MulIr, Irreps, str)):
            raise NotImplementedError('x.at[..., "1e + 2e"] is not implemented')

        # Support of x[..., 3:32]
        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and index[-1].step is None
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            raise NotImplementedError("x.at[..., 3:32] is not implemented")

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                raise IndexError(f"Indexing with {index[-1]} in the irreps dimension is not supported.")

        # Support of x.at[index, :].add(IrrArray(...))
        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError("The irreps of the array and the values to add must be the same.")

            values = values.rechunk(self.irreps)

            zero_flags = tuple(x and y for x, y in zip(self.zero_flags, values.zero_flags))
            return IrrepsArray(
                self.irreps,
                self.array.at[index].add(values.array),
                zero_flags=zero_flags,
            )

        raise NotImplementedError(f"x.at[i].add(v) with v={type(values)} is not implemented.")


class _MulIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_mul only supports one slices (like IrrepsArray.slice_by_mul[2:4]).")
        start, stop, stride = index.indices(self.irreps_array.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("IrrepsArray.slice_by_mul does not support strides.")

        irreps = []
        list = []
        i = 0
        for (mul, ir), x in zip(self.irreps_array.irreps, self.irreps_array.chunks):
            if start <= i and i + mul <= stop:
                irreps.append((mul, ir))
                list.append(x)
            elif start < i + mul and i < stop:
                irreps.append((min(stop, i + mul) - max(start, i), ir))
                list.append(x[..., max(start, i) - i : min(stop, i + mul) - i, :])

            i += mul
        return o3.experimental.from_chunks(
            irreps,
            list,
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
        )


class _DimIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_dim only supports slices (like IrrepsArray.slice_by_dim[2:4]).")
        return self.irreps_array[..., index]


class _ChunkIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_chunk only supports slices (like IrrepsArray.slice_by_chunk[2:4]).")
        start, stop, stride = index.indices(len(self.irreps_array.irreps))

        return o3.experimental.from_chunks(
            self.irreps_array.irreps[start:stop:stride],
            self.irreps_array.chunks[start:stop:stride],
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
        )
