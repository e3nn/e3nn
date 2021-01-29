from math import sqrt
from collections import namedtuple

import torch
from e3nn import o3
from e3nn.util import eval_code, broadcast_tensors


def _prod(x):
    out = 1
    for a in x:
        out *= a
    return out


class TensorProduct(torch.nn.Module):
    r"""Tensor Product with parametrizable paths

    Parameters
    ----------
    in1 : `Irreps` or list of tuple
        List of first inputs ``(multiplicity, irrep[, variance])``.

    in2 : `Irreps` or list of tuple
        List of second inputs ``(multiplicity, irrep[, variance])``.

    out : `Irreps` or list of tuple
        List of outputs ``(multiplicity, irrep[, variance])``.

    instructions : list of tuple
        List of instructions ``(i_1, i_2, i_out, mode, train[, path_weight])``
        it means: Put ``in1[i_1]`` :math:`\otimes` ``in2[i_2]`` into ``out[i_out]``

        * mode: determines the way the multiplicities are treated, "uvw" is fully connected
        * train: `True` of `False` if this path is weighed by a parameter
        * path weight: how much this path should contribute to the output

    normalization : {'component', 'norm'}
        the way it is assumed the representation are normalized. If it is set to "norm":

        .. math::

            \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1

    internal_weights : bool
        does the instance of the class contains the parameters

    shared_weights : bool
        are the parameters shared among the inputs extra dimensions

        * `True` :math:`z_i = w x_i \otimes y_i`
        * `False` :math:`z_i = w_i x_i \otimes y_i`

        where here :math:`i` denotes a *batch-like* index

    Examples
    --------
    Create a module that computes elementwise the cross-product of 16 vectors with 16 vectors :math:`z_u = x_u \wedge y_u`

    >>> module = TensorProduct(
    ...     "16x1o", "16x1o", "16x1e",
    ...     [
    ...         (0, 0, 0, "uuu", False)
    ...     ]
    ... )

    Now mix all 16 vectors with all 16 vectors to makes 16 pseudo-vectors :math:`z_w = \sum_{u,v} w_{uvw} x_u \wedge y_v`

    >>> module = TensorProduct(
    ...     [(16, (1, -1))],
    ...     [(16, (1, -1))],
    ...     [(16, (1,  1))],
    ...     [
    ...         (0, 0, 0, "uvw", True)
    ...     ]
    ... )

    With custom input variance and custom path weights:

    >>> module = TensorProduct(
    ...     "8x0o + 8x1o",
    ...     [(16, "1o", 1/16)],
    ...     "16x1e",
    ...     [
    ...         (0, 0, 0, "uvw", True, 3),
    ...         (1, 0, 0, "uvw", True, 1),
    ...     ]
    ... )

    Example of a dot product:

    >>> irreps = o3.Irreps("3x0e + 4x0o + 1e + 2o + 3o")
    >>> module = TensorProduct(irreps, irreps, "0e", [
    ...     (i, i, 0, 'uuw', False)
    ...     for i, (mul, ir) in enumerate(irreps)
    ... ])

    Implement :math:`z_u = x_u \otimes (\sum_v w_{uv} y_v)`

    >>> module = TensorProduct(
    ...     "8x0o + 7x1o + 3x2e",
    ...     "10x0e + 10x1e + 10x2e",
    ...     "8x0o + 7x1o + 3x2e",
    ...     [
    ...         # paths for the l=0:
    ...         (0, 0, 0, "uvu", True),  # 0x0->0
    ...         # paths for the l=1:
    ...         (1, 0, 1, "uvu", True),  # 1x0->1
    ...         (1, 1, 1, "uvu", True),  # 1x1->1
    ...         (1, 2, 1, "uvu", True),  # 1x2->1
    ...         # paths for the l=2:
    ...         (2, 0, 2, "uvu", True),  # 2x0->2
    ...         (2, 1, 2, "uvu", True),  # 2x1->2
    ...         (2, 2, 2, "uvu", True),  # 2x2->2
    ...     ]
    ... )

    Tensor Product using the xavier uniform initialization:

    >>> irreps_1 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_2 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_out = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> # create a Fully Connected Tensor Product
    >>> module = o3.TensorProduct(
    ...     irreps_1,
    ...     irreps_2,
    ...     irreps_out,
    ...     [
    ...         (i_1, i_2, i_out, "uvw", True, mul_1 * mul_2)
    ...         for i_1, (mul_1, ir_1) in enumerate(irreps_1)
    ...         for i_2, (mul_2, ir_2) in enumerate(irreps_2)
    ...         for i_out, (mul_out, ir_out) in enumerate(irreps_out)
    ...         if ir_out in ir_1 * ir_2
    ...     ]
    ... )
    >>> with torch.no_grad():
    ...     for weight in module.parameters():
    ...         # formula from torch.nn.init.xavier_uniform_
    ...         mul_1, mul_2, mul_out = weight.shape
    ...         a = (6 / (mul_1 * mul_2 + mul_out))**0.5
    ...         _ = weight.uniform_(-a, a)  # `_ = ` is only here because of pytest
    >>> n = 1_000
    >>> vars = module(irreps_1.randn(n, -1), irreps_2.randn(n, -1)).var(0)
    >>> assert vars.min() > 1 / 3
    >>> assert vars.max() < 3
    """
    def __init__(
            self,
            in1,
            in2,
            out,
            instructions,
            normalization='component',
            internal_weights=None,
            shared_weights=None,
            _specialized_code=True,
                ):
        super().__init__()

        assert normalization in ['component', 'norm'], normalization

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights

        try:
            in1 = o3.Irreps(in1)
        except AssertionError:
            pass
        try:
            in2 = o3.Irreps(in2)
        except AssertionError:
            pass
        try:
            out = o3.Irreps(out)
        except AssertionError:
            pass

        in1 = [x if len(x) == 3 else x + (1.0,) for x in in1]
        in2 = [x if len(x) == 3 else x + (1.0,) for x in in2]
        out = [x if len(x) == 3 else x + (1.0,) for x in out]

        self.irreps_in1 = o3.Irreps([(mul, ir) for mul, ir, _var in in1])
        self.irreps_in2 = o3.Irreps([(mul, ir) for mul, ir, _var in in2])
        self.irreps_out = o3.Irreps([(mul, ir) for mul, ir, _var in out])

        in1_var = [var for _, _, var in in1]
        in2_var = [var for _, _, var in in2]
        out_var = [var for _, _, var in out]

        self.shared_weights = shared_weights
        z = '' if self.shared_weights else 'z'

        code_out = f"""
from typing import List

import torch

@torch.jit.script
def main(x1: torch.Tensor, x2: torch.Tensor, ws: List[torch.Tensor], w3j: List[torch.Tensor]) -> torch.Tensor:
    batch = x1.shape[0]
    out = x1.new_zeros((batch, {self.irreps_out.dim}))
    ein = torch.einsum
"""

        code_right = f"""
from typing import List

import torch

@torch.jit.script
def main(x2: torch.Tensor, ws: List[torch.Tensor], w3j: List[torch.Tensor]) -> torch.Tensor:
    batch = x2.shape[0]
    out = x2.new_zeros((batch, {self.irreps_in1.dim}, {self.irreps_out.dim}))
    ein = torch.einsum
"""
        s = 4 * " "

        wshapes = []
        wigners = []

        for i_1, (mul_1, (l_1, p_1)) in enumerate(self.irreps_in1):
            index_1 = self.irreps_in1[:i_1].dim
            dim_1 = mul_1 * (2 * l_1 + 1)
            code_out += f"{s}x1_{i_1} = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})\n"
        code_out += "\n"

        for i_2, (mul_2, (l_2, p_2)) in enumerate(self.irreps_in2):
            index_2 = self.irreps_in2[:i_2].dim
            dim_2 = mul_2 * (2 * l_2 + 1)
            line = f"{s}x2_{i_2} = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})\n"
            code_out += line
            code_right += line
        code_out += "\n"
        code_right += "\n"

        last_ss = None

        Instruction = namedtuple("Instruction", "i_in1, i_in2, i_out, connection_mode, has_weight, path_weight")
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        self.instructions = [
            Instruction(i_1, i_2, i_out, mode, weight, path_weight)
            for i_1, i_2, i_out, mode, weight, path_weight in instructions
        ]

        for i_1, i_2, i_out, mode, weight, path_weight in self.instructions:
            mul_1, (l_1, p_1) = self.irreps_in1[i_1]
            mul_2, (l_2, p_2) = self.irreps_in2[i_2]
            mul_out, (l_out, p_out) = self.irreps_out[i_out]
            dim_1 = mul_1 * (2 * l_1 + 1)
            dim_2 = mul_2 * (2 * l_2 + 1)
            dim_out = mul_out * (2 * l_out + 1)
            index_1 = self.irreps_in1[:i_1].dim
            index_2 = self.irreps_in2[:i_2].dim
            index_out = self.irreps_out[:i_out].dim

            assert p_1 * p_2 == p_out
            assert abs(l_1 - l_2) <= l_out <= l_1 + l_2

            if dim_1 == 0 or dim_2 == 0 or dim_out == 0:
                continue

            alpha = path_weight * out_var[i_out] / sum(in1_var[i_1_] * in2_var[i_2_] for i_1_, i_2_, i_out_, _, _, _ in self.instructions if i_out_ == i_out)

            s = 4 * " "

            line = (
                f"{s}with torch.autograd.profiler.record_function("
                f"'{self.irreps_in1[i_1:i_1+1]} x {self.irreps_in2[i_2:i_2+1]} "
                f"= {self.irreps_out[i_out:i_out+1]} {mode} {weight}'):\n"
            )
            code_out += line
            code_right += line

            s = 8 * " "

            code_out += f"{s}s1 = x1_{i_1}\n"
            code_right += f"{s}e1 = torch.eye({mul_1}, dtype=x2.dtype, device=x2.device)\n"

            line = f"{s}s2 = x2_{i_2}\n"
            code_out += line
            code_right += line

            assert mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            alpha = sqrt(alpha / {
                'uvw': (mul_1 * mul_2),
                'uvu': mul_2,
                'uvv': mul_1,
                'uuw': mul_1,
                'uuu': 1,
                'uvuv': 1,
            }[mode])

            index_w = len(wshapes)
            if weight:
                wshapes.append({
                    'uvw': (mul_1, mul_2, mul_out),
                    'uvu': (mul_1, mul_2),
                    'uvv': (mul_1, mul_2),
                    'uuw': (mul_1, mul_out),
                    'uuu': (mul_1,),
                    'uvuv': (mul_1, mul_2),
                }[mode])

            line_out = f"{s}out[:, {index_out}:{index_out+dim_out}] += {alpha} * {{}}.reshape(batch, {dim_out})\n\n"
            line_right = f"{s}out[:, {index_1}:{index_1+dim_1}, {index_out}:{index_out+dim_out}] += {alpha} * {{}}.reshape(batch, {dim_1}, {dim_out})\n\n"

            if _specialized_code:
                # optimized code for special cases:
                # 0 x 0 = 0
                # 0 x L = L
                # L x 0 = L
                # L x L = 0
                # 1 x 1 = 1

                if (l_1, l_2, l_out) == (0, 0, 0) and mode in ['uvw', 'uvu'] and normalization in ['component', 'norm'] and weight:
                    code_out += f"{s}s1 = s1.reshape(batch, {mul_1})\n"
                    line = f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                    code_out += line
                    code_right += line

                    if mode == 'uvw':
                        code_out += line_out.format(f"ein('{z}uvw,zu,zv->zw', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uvw,zv->zuw', ws[{index_w}], s2)")
                    if mode == 'uvu':
                        code_out += line_out.format(f"ein('{z}uv,zu,zv->zu', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uv,uw,zv->zuw', ws[{index_w}], e1, s2)")

                    continue

                if l_1 == 0 and l_2 == l_out and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code_out += f"{s}s1 = s1.reshape(batch, {mul_1})\n"

                    if mode == 'uvw':
                        code_out += line_out.format(f"ein('{z}uvw,zu,zvi->zwi', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uvw,zvi->zuwi', ws[{index_w}], s2)")
                    if mode == 'uvu':
                        code_out += line_out.format(f"ein('{z}uv,zu,zvi->zui', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uv,uw,zvi->zuwi', ws[{index_w}], e1, s2)")

                    continue

                if l_1 == l_out and l_2 == 0 and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code_out += f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                    code_right += f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                    code_right += f"{s}wig = torch.eye({2 * l_1 + 1}, dtype=x2.dtype, device=x2.device)\n"

                    if mode == 'uvw':
                        code_out += line_out.format(f"ein('{z}uvw,zui,zv->zwi', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uvw,ij,zv->zuiwj', ws[{index_w}], wig, s2)")
                    if mode == 'uvu':
                        code_out += line_out.format(f"ein('{z}uv,zui,zv->zui', ws[{index_w}], s1, s2)")
                        code_right += line_right.format(f"ein('{z}uv,ij,uw,zv->zuiwj', ws[{index_w}], wig, e1, s2)")

                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvw' and normalization == 'component' and weight:
                    # Cl_l_0 = eye / sqrt(2L+1)
                    code_out += line_out.format(f"ein('{z}uvw,zui,zvi->zw', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uvw,zvi->zuiw', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, s2)")
                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvu' and normalization == 'component' and weight:
                    # Cl_l_0 = eye / sqrt(2L+1)
                    code_out += line_out.format(f"ein('{z}uv,zui,zvi->zu', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uv,uw,zvi->zuiw', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, e1, s2)")
                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uuu' and normalization == 'component' and weight:
                    # Cl_l_0 = eye / sqrt(2L+1)
                    code_out += line_out.format(f"ein('{z}u,zui,zui->zu', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}u,uw,zui->zuiw', ws[{index_w}] / {sqrt(2 * l_1 + 1)}, e1, s2)")
                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uuu' and normalization == 'component' and not weight:
                    # Cl_l_0 = eye / sqrt(2L+1)
                    code_out += line_out.format(f"ein('zui,zui->zu', s1, s2).div({sqrt(2 * l_1 + 1)})")
                    code_right += line_right.format(f"ein('uw,zui->zuiw', e1, s2).div({sqrt(2 * l_1 + 1)})")
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvw' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code_out += f"{s}s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code_out += f"{s}s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code_out += f"{s}s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code_out += line_out.format(f"ein('{z}uvw,zuvi->zwi', ws[{index_w}] / {sqrt(2)}, torch.cross(s1, s2, dim=3))")

                    if (l_1, l_2, l_out) in wigners:
                        index_w3j = wigners.index((l_1, l_2, l_out))
                    else:
                        index_w3j = len(wigners)
                        wigners += [(l_1, l_2, l_out)]

                    code_right += line_right.format(f"ein('{z}uvw,ijk,zvj->zuiwk', ws[{index_w}], w3j[{index_w3j}], s2)")
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvu' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code_out += f"{s}s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code_out += f"{s}s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code_out += f"{s}s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code_out += line_out.format(f"ein('{z}uv,zuvi->zui', ws[{index_w}] / {sqrt(2)}, torch.cross(s1, s2, dim=3))")

                    if (l_1, l_2, l_out) in wigners:
                        index_w3j = wigners.index((l_1, l_2, l_out))
                    else:
                        index_w3j = len(wigners)
                        wigners += [(l_1, l_2, l_out)]

                    code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwk', ws[{index_w}], w3j[{index_w3j}], e1, s2)")
                    continue

            if last_ss != (i_1, i_2, mode[:2]):
                if mode[:2] == 'uv':
                    code_out += f"{s}ss = ein('zui,zvj->zuvij', s1, s2)\n"
                if mode[:2] == 'uu':
                    code_out += f"{s}ss = ein('zui,zuj->zuij', s1, s2)\n"
                last_ss = (i_1, i_2, mode[:2])

            if (l_1, l_2, l_out) in wigners:
                index_w3j = wigners.index((l_1, l_2, l_out))
            else:
                index_w3j = len(wigners)
                wigners += [(l_1, l_2, l_out)]

            if mode == 'uvw':
                assert weight
                code_out += line_out.format(f"ein('{z}uvw,ijk,zuvij->zwk', ws[{index_w}], w3j[{index_w3j}], ss)")
                code_right += line_right.format(f"ein('{z}uvw,ijk,zvj->zuiwk', ws[{index_w}], w3j[{index_w3j}], s2)")
            if mode == 'uvu':
                assert mul_1 == mul_out
                if weight:
                    code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zuk', ws[{index_w}], w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwk', ws[{index_w}], w3j[{index_w3j}], e1, s2)")
                else:
                    code_out += line_out.format(f"ein('ijk,zuvij->zuk', w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('ijk,uw,zvj->zuiwk', w3j[{index_w3j}], e1, s2)")
            if mode == 'uvv':
                assert mul_2 == mul_out
                if weight:
                    code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zvk', ws[{index_w}], w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('{z}uv,ijk,zvj->zuivk', ws[{index_w}], w3j[{index_w3j}], s2)")
                else:
                    code_out += line_out.format(f"ein('ijk,zuvij->zvk', w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('u,ijk,zvj->zuivk', s2.new_zeros({mul_1}).fill_(1.0), w3j[{index_w3j}], s2)")
            if mode == 'uuw':
                assert mul_1 == mul_2
                if weight:
                    code_out += line_out.format(f"ein('{z}uw,ijk,zuij->zwk', ws[{index_w}], w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('{z}uw,ijk,zuj->zuiwk', ws[{index_w}], w3j[{index_w3j}], s2)")
                else:
                    assert mul_out == 1
                    code_out += line_out.format(f"ein('ijk,zuij->zk', w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('ijk,zuj->zuik', w3j[{index_w3j}], s2)")
            if mode == 'uuu':
                assert mul_1 == mul_2 == mul_out
                if weight:
                    code_out += line_out.format(f"ein('{z}u,ijk,zuij->zuk', ws[{index_w}], w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('{z}u,ijk,uw,zuj->zuiwk', ws[{index_w}], w3j[{index_w3j}], e1, s2)")
                else:
                    code_out += line_out.format(f"ein('ijk,zuij->zuk', w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('ijk,uw,zuj->zuiwk', w3j[{index_w3j}], e1, s2)")
            if mode == 'uvuv':
                assert mul_1 * mul_2 == mul_out
                if weight:
                    code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zuvk', ws[{index_w}], w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwvk', ws[{index_w}], w3j[{index_w3j}], e1, s2)")
                else:
                    code_out += line_out.format(f"ein('ijk,zuvij->zuvk', w3j[{index_w3j}], ss)")
                    code_right += line_right.format(f"ein('ijk,uw,zvj->zuiwvk', w3j[{index_w3j}], e1, s2)")
            code_out += "\n"

        s = 4 * " "
        code_out += f"{s}return out"
        code_right += f"{s}return out"

        self.code_out = code_out
        self.code_right = code_right

        # w3j
        self.wigners = wigners
        for i, (l_1, l_2, l_out) in enumerate(self.wigners):
            wig = o3.wigner_3j(l_1, l_2, l_out)

            if normalization == 'component':
                wig *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

            self.register_buffer(f"C{i}", wig)

        # weights
        self.weight_shapes = wshapes
        self.weight_numel = sum(_prod(shape) for shape in self.weight_shapes)
        self.weight_infos = [
            (i_1, i_2, i_out, mode, path_weight, shape)
            for (i_1, i_2, i_out, mode, path_weight), shape in zip(
                [
                    (i_1, i_2, i_out, mode, path_weight)
                    for i_1, i_2, i_out, mode, weight, path_weight in self.instructions
                    if weight
                ],
                wshapes
            )
        ]

        if internal_weights:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.ParameterDict()
            for i_1, i_2, i_out, mode, path_weight, shape in self.weight_infos:
                name = f'[{i_1}:{self.irreps_in1[i_1:i_1+1]}] x [{i_2}:{self.irreps_in2[i_2:i_2+1]}] -> [{i_out}:{self.irreps_out[i_out:i_out+1]}]'
                self.weight[name] = torch.nn.Parameter(torch.randn(shape))

        self.to(dtype=torch.get_default_dtype())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {self.weight_numel} weights)"
        )

    def prepare_weight_list(self, weight):
        if self.weight_numel:
            if weight is None:
                weight = list(self.weight.values())
            if torch.is_tensor(weight):
                ws = []
                i = 0
                for shape in self.weight_shapes:
                    d = _prod(shape)
                    if not self.shared_weights:
                        ws += [weight[..., i:i+d].reshape((-1,) + shape)]
                    else:
                        ws += [weight[i:i+d].reshape(shape)]
                    i += d
                weight = ws
            if isinstance(weight, list):
                if not self.shared_weights:
                    weight = [w.reshape(-1, *shape) for w, shape in zip(weight, self.weight_shapes)]
                else:
                    weight = [w.reshape(*shape) for w, shape in zip(weight, self.weight_shapes)]
        else:
            weight = []
        return weight

    def right(self, features_2, weight=None):
        r"""evaluate partially :math:`w x \cdot \otimes y`

        It returns an operator in the form of a matrix.

        Parameters
        ----------
        features_2 : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``self.weight_shapes`` / ``(...) + self.weight_shapes``.
            Use ``self.weight_infos`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim, irreps_out.dim)``
        """
        with torch.autograd.profiler.record_function(repr(self)):
            size = features_2.shape[:-1]
            assert features_2.shape[-1] == self.irreps_in2.dim, f"{features_2.shape} is not (..., {self.irreps_in2.dim})"
            features_2 = features_2.reshape(_prod(size), self.irreps_in2.dim)

            if features_2.shape[0] == 0:
                return torch.zeros(*size, self.irreps_in1.dim, self.irreps_out.dim)

            weight = self.prepare_weight_list(weight)
            wigners = [getattr(self, f"C{i}") for i in range(len(self.wigners))]

            operator = eval_code(self.code_right).main(features_2, weight, wigners)

            return operator.reshape(*size, self.irreps_in1.dim, self.irreps_out.dim)

    def forward(self, features_1, features_2, weight=None):
        r"""evaluate :math:`w x \otimes y`

        Parameters
        ----------
        features_1 : `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim)``

        features_2 : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``self.weight_shapes`` / ``(...) + self.weight_shapes``.
            Use ``self.weight_infos`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        with torch.autograd.profiler.record_function(repr(self)):
            features_1, features_2 = broadcast_tensors(features_1, features_2)
            size = features_1.shape[:-1]
            assert features_1.shape[-1] == self.irreps_in1.dim, f"{features_1.shape} is not (..., {self.irreps_in1.dim})"
            assert features_2.shape[-1] == self.irreps_in2.dim, f"{features_2.shape} is not (..., {self.irreps_in2.dim})"
            features_1 = features_1.reshape(_prod(size), self.irreps_in1.dim)
            features_2 = features_2.reshape(_prod(size), self.irreps_in2.dim)

            if features_1.shape[0] == 0:
                return torch.zeros(*size, self.irreps_out.dim)

            weight = self.prepare_weight_list(weight)
            wigners = [getattr(self, f"C{i}") for i in range(len(self.wigners))]

            features = eval_code(self.code_out).main(features_1, features_2, weight, wigners)

            return features.reshape(*size, self.irreps_out.dim)


class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicites.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    irreps_out : `Irreps`
        representation of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`

    internal_weights : bool
        see `TensorProduct`

    shared_weights : bool
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            irreps_out,
            normalization='component',
            internal_weights=None,
            shared_weights=None
                ):
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()

        in1 = [(mul, ir, 1.0) for mul, ir in irreps_in1]
        in2 = [(mul, ir, 1.0) for mul, ir in irreps_in2]
        out = [(mul, ir, 1.0) for mul, ir in irreps_out]

        instr = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, (l_1, p_1)) in enumerate(irreps_in1)
            for i_2, (_, (l_2, p_2)) in enumerate(irreps_in2)
            for i_out, (_, (l_out, p_out)) in enumerate(irreps_out)
            if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
        ]
        super().__init__(in1, in2, out, instr, normalization, internal_weights, shared_weights)


class ElementwiseTensorProduct(TensorProduct):
    r"""Elementwise-Connected tensor product

    .. math::

        z_u = x_u \otimes y_u

    where :math:`u` runs over the irrep note that ther is no weights.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    irreps_out : iterator of `Irrep`, optional
        representations of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            irreps_out=None,
            normalization='component',
                ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if irreps_out is not None:
            irreps_out = [o3.Irrep(ir) for ir in irreps_out]

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

                if irreps_out is not None and ir not in irreps_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [
                    (i, i, i_out, 'uuu', False)
                ]

        super().__init__(irreps_in1, irreps_in2, out, instr, normalization, internal_weights=False)


class FullTensorProduct(TensorProduct):
    r"""Full tensor product between two irreps

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` runs over the irrep, note that ther is no weights.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    irreps_out : iterator of `Irrep`, optional
        representations of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            irreps_out=None,
            normalization='component',
                ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if irreps_out is not None:
            irreps_out = [o3.Irrep(ir) for ir in irreps_out]

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:

                    if irreps_out is not None and ir_out not in irreps_out:
                        continue

                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr += [
                        (i_1, i_2, i_out, 'uvuv', False)
                    ]

        out = o3.Irreps(out)
        out, p, _ = out.sort()

        instr = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instr
        ]

        super().__init__(irreps_in1, irreps_in2, out, instr, normalization, internal_weights=False)


class Linear(TensorProduct):
    r"""Linear operation equivariant to :math:`O(3)`

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    irreps_out : `Irreps`
        representation of the output

    internal_weights : bool
        see `TensorProduct`

    shared_weights : bool
        see `TensorProduct`

    Examples
    --------
    Linearly combines 4 scalars into 8 scalars and 16 vectors into 8 vectors.

    >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
    >>> lin.weight_numel
    160
    """
    def __init__(
            self,
            irreps_in,
            irreps_out,
            internal_weights=None,
            shared_weights=None,
                ):
        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()

        instr = [
            (i_in, 0, i_out, 'uvw', True, 1.0)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        super().__init__(irreps_in, "0e", irreps_out, instr, internal_weights=internal_weights, shared_weights=shared_weights)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        output_mask = torch.cat([
            torch.ones(mul * (2 * l + 1))
            if any(l_in == l and p_in == p for _, (l_in, p_in) in self.irreps_in)
            else torch.zeros(mul * (2 * l + 1))
            for mul, (l, p) in self.irreps_out
        ])
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(self, features, weight=None):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        weight : `torch.Tensor`, optional
            required if ``internal_weights`` is `False`

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        ones = features.new_ones(features.shape[:-1] + (1,))
        return super().forward(features, ones, weight)


class Norm(TensorProduct):
    r"""Norm operation

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    normalization : {'component', 'norm'}
        see `TensorProduct`

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(torch.randn(17 * 3)).shape
    torch.Size([17])
    """
    def __init__(
            self,
            irreps_in,
                ):
        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, 'uuu', False, ir.dim)
            for i, (mul, ir) in enumerate(irreps_in)
        ]

        super().__init__(irreps_in, irreps_in, irreps_out, instr, 'component')

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, features):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        return super().forward(features, features).sqrt()
