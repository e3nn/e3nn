from typing import Iterator, Optional

import torch
from e3nn import o3
from e3nn.util import prod

from ._tensor_product import TensorProduct


class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicities.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`

    path_normalization : {'element', 'path'}
        see `e3nn.o3.TensorProduct`

    internal_weights : bool
        see `e3nn.o3.TensorProduct`

    shared_weights : bool
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self, irreps_in1, irreps_in2, irreps_out, irrep_normalization: str = None, path_normalization: str = None, **kwargs
    ) -> None:
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )


class ElementwiseTensorProduct(TensorProduct):
    r"""Elementwise connected tensor product.

    .. math::

        z_u = x_u \otimes y_u

    where :math:`u` runs over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`

    Examples
    --------
    Elementwise scalar product

    >>> ElementwiseTensorProduct("5x1o + 5x1e", "10x1e", ["0e", "0o"])
    ElementwiseTensorProduct(5x1o+5x1e x 10x1e -> 5x0o+5x0e | 10 paths | 0 weights)

    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, irrep_normalization: str = None, **kwargs) -> None:
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep")

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instr = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [(i, i, i_out, "uuu", False)]

        super().__init__(irreps_in1, irreps_in2, out, instr, irrep_normalization=irrep_normalization, **kwargs)


class FullTensorProduct(TensorProduct):
    r"""Full tensor product between two irreps.

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` run over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        filter_ir_out: Iterator[o3.Irrep] = None,
        irrep_normalization: str = None,
        **kwargs,
    ) -> None:
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep")

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue

                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr += [(i_1, i_2, i_out, "uvuv", False)]

        out = o3.Irreps(out)
        out, p, _ = out.sort()

        instr = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instr]

        super().__init__(irreps_in1, irreps_in2, out, instr, irrep_normalization=irrep_normalization, **kwargs)


class ChannelWiseTensorProduct(TensorProduct):
    r"""Nequip-like TensorProduct with weights.

    .. math::

        z_{u} = w_{u} x_u \otimes y

    where :math:`u` runs over the irreps.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        representation of the first input

    irreps_in2 : `e3nn.o3.Irreps`
        representation of the second input

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`

    path_normalization : {'element', 'path'}
        see `e3nn.o3.TensorProduct`

    internal_weights : bool
        see `e3nn.o3.TensorProduct`

    shared_weights : bool
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        irrep_normalization: str = None,
        path_normalization: str = None,
        **kwargs,
    ) -> None:
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()

        # Borrowed from https://github.com/mir-group/nequip/blob/1e150cdc8614e640116d11e085d8e5e45b21e94d/nequip/nn/_interaction_block.py#L83-L112
        out = []
        instr = []
        for i_1, (mul, ir_1) in enumerate(irreps_in1):
            for i_2, (_, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if ir_out in irreps_out:
                        i_out = len(out)
                        out.append((mul, ir_out))
                        instr += [(i_1, i_2, i_out, "uvu", True)]

        out = o3.Irreps(out)
        out, p, _ = out.sort()

        instr = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instr]

        super().__init__(irreps_in1, irreps_in2, out, instr, irrep_normalization=irrep_normalization, **kwargs)


def _square_instructions_full(irreps_in, filter_ir_out=None, irrep_normalization=None):
    """Generate instructions for square tensor product.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm', 'none'}
        see `e3nn.o3.TensorProduct`

    Returns
    -------
    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    instr : list of tuple
        list of instructions

    """
    # pylint: disable=too-many-nested-blocks
    irreps_out = []
    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in):
            for ir_out in ir_1 * ir_2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                if irrep_normalization == "component":
                    alpha = ir_out.dim
                if irrep_normalization == "norm":
                    alpha = ir_1.dim * ir_2.dim
                if irrep_normalization == "none":
                    alpha = 1

                if i_1 < i_2:
                    i_out = len(irreps_out)
                    irreps_out.append((mul_1 * mul_2, ir_out))
                    instr += [(i_1, i_2, i_out, "uvuv", False, alpha)]
                elif i_1 == i_2:
                    i = i_1
                    mul = mul_1

                    if mul > 1:
                        i_out = len(irreps_out)
                        irreps_out.append((mul * (mul - 1) // 2, ir_out))
                        instr += [(i, i, i_out, "uvu<v", False, alpha)]

                    if ir_out.l % 2 == 0:
                        if irrep_normalization == "component":
                            if ir_out.l == 0:
                                alpha = ir_out.dim / (ir_1.dim + 2)
                            else:
                                alpha = ir_out.dim / 2
                        if irrep_normalization == "norm":
                            if ir_out.l == 0:
                                alpha = ir_out.dim * ir_1.dim
                            else:
                                alpha = ir_1.dim * (ir_1.dim + 2) / 2

                        i_out = len(irreps_out)
                        irreps_out.append((mul, ir_out))
                        instr += [(i, i, i_out, "uuu", False, alpha)]

    irreps_out = o3.Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instr = [(i_1, i_2, p[i_out], mode, train, alpha) for i_1, i_2, i_out, mode, train, alpha in instr]

    return irreps_out, instr


def _square_instructions_fully_connected(irreps_in, irreps_out, irrep_normalization=None):
    """Generate instructions for square tensor product.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm', 'none'}
        see `e3nn.o3.TensorProduct`

    Returns
    -------
    instr : list of tuple
        list of instructions
    """
    # pylint: disable=too-many-nested-blocks
    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (_mul_2, ir_2) in enumerate(irreps_in):
            for i_out, (_mul_out, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:
                    if irrep_normalization == "component":
                        alpha = ir_out.dim
                    if irrep_normalization == "norm":
                        alpha = ir_1.dim * ir_2.dim
                    if irrep_normalization == "none":
                        alpha = 1

                    if i_1 < i_2:
                        instr += [(i_1, i_2, i_out, "uvw", True, alpha)]
                    elif i_1 == i_2:
                        i = i_1
                        mul = mul_1

                        if mul > 1:
                            instr += [(i, i, i_out, "u<vw", True, alpha)]

                        if ir_out.l % 2 == 0:
                            if irrep_normalization == "component":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_out.dim / 2
                            if irrep_normalization == "norm":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2

                            instr += [(i, i, i_out, "uuw", True, alpha)]

    return instr


class TensorSquare(TensorProduct):
    r"""Compute the square tensor product of a tensor and reduce it in irreps

    If `irreps_out` is given, this operation is fully connected.
    If `irreps_out` is not given, the operation has no parameter and is like full tensor product.

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    irreps_out : `e3nn.o3.Irreps`, optional
        representation of the output

    filter_ir_out : iterator of `e3nn.o3.Irrep`, optional
        filter to select only specific `e3nn.o3.Irrep` of the output

    irrep_normalization : {'component', 'norm'}
        see `e3nn.o3.TensorProduct`
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps = None,
        filter_ir_out: Iterator[o3.Irrep] = None,
        irrep_normalization: str = None,
        **kwargs,
    ) -> None:
        if irrep_normalization is None:
            irrep_normalization = "component"

        assert irrep_normalization in ["component", "norm", "none"]

        irreps_in = o3.Irreps(irreps_in).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError as exc:
                raise ValueError(f"Error constructing filter_ir_out irrep: {exc}") from exc

        if irreps_out is None:
            irreps_out, instr = _square_instructions_full(irreps_in, filter_ir_out, irrep_normalization)
        else:
            if filter_ir_out is not None:
                raise ValueError("Both `irreps_out` and `filter_ir_out` are not None, this is ambiguous.")

            irreps_out = o3.Irreps(irreps_out).simplify()

            instr = _square_instructions_fully_connected(irreps_in, irreps_out, irrep_normalization)

        self.irreps_in = irreps_in

        super().__init__(irreps_in, irreps_in, irreps_out, instr, irrep_normalization="none", **kwargs)

    def __repr__(self) -> str:
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )

    def forward(self, x, weight: Optional[torch.Tensor] = None):  # pylint: disable=arguments-differ
        return super().forward(x, x, weight)
