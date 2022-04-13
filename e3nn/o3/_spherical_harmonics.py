r"""Spherical Harmonics as polynomials of x, y, z
"""
from typing import Union, List, Any

import math

import sympy
from sympy.printing.pycode import pycode
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode('script')
class SphericalHarmonics(torch.nn.Module):
    """JITable module version of :meth:`e3nn.o3.spherical_harmonics`.

    Parameters are identical to :meth:`e3nn.o3.spherical_harmonics`.
    """
    normalize: bool
    normalization: str
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str

    def __init__(
        self,
        irreps_out: Union[int, List[int], str, o3.Irreps],
        normalize: bool,
        normalization: str = 'integral',
        irreps_in: Any = None,
    ):
        super().__init__()
        self.normalize = normalize
        self.normalization = normalization
        assert normalization in ['integral', 'component', 'norm']

        if isinstance(irreps_out, str):
            irreps_out = o3.Irreps(irreps_out)
        if isinstance(irreps_out, o3.Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = o3.Irreps("1e")
        if irreps_in is None:
            irreps_in = o3.Irreps("1o")

        irreps_in = o3.Irreps(irreps_in)
        if irreps_in not in (o3.Irreps("1x1o"), o3.Irreps("1x1e")):
            raise ValueError(f"irreps_in for SphericalHarmonics must be either a vector (`1x1o`) or a pseudovector (`1x1e`), not `{irreps_in}`")
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p  # pylint: disable=no-member

        if isinstance(irreps_out, o3.Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p**l:
                    raise ValueError(f"irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {l} with parity p = {p}, which is inconsistent with the input parity {input_p} — the output parity should have been p = {input_p**l}")
                ls.extend([l]*mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        irreps_out = o3.Irreps([(1, (l, input_p**l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f'spherical_harmonics({ls})'

        _lmax = 11
        if self._lmax > _lmax:
            raise NotImplementedError(f'spherical_harmonics maximum l implemented is {_lmax}, send us an email to ask for more')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # - PROFILER - with torch.autograd.profiler.record_function(self._prof_str):
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)  # forward 0's instead of nan for zero-radius

        sh = _spherical_harmonics(self._lmax, x[..., 0], x[..., 1], x[..., 2])

        if not self._is_range_lmax:
            sh = torch.cat([
                sh[..., l*l:(l+1)*(l+1)]
                for l in self._ls_list
            ], dim=-1)

        if self.normalization == 'integral':
            sh.mul_(torch.cat([
                (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
                for l in self._ls_list
            ]))
        elif self.normalization == 'component':
            sh.mul_(torch.cat([
                math.sqrt(2 * l + 1) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
                for l in self._ls_list
            ]))

        return sh


def spherical_harmonics(
    l: Union[int, List[int], str, o3.Irreps],
    x: torch.Tensor,
    normalize: bool,
    normalization: str = 'integral'
):
    r"""Spherical harmonics

    .. image:: https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif

    | Polynomials defined on the 3d space :math:`Y^l: \mathbb{R}^3 \longrightarrow \mathbb{R}^{2l+1}`
    | Usually restricted on the sphere (with ``normalize=True``) :math:`Y^l: S^2 \longrightarrow \mathbb{R}^{2l+1}`
    | who satisfies the following properties:

    * are polynomials of the cartesian coordinates ``x, y, z``
    * is equivariant :math:`Y^l(R x) = D^l(R) Y^l(x)`
    * are orthogonal :math:`\int_{S^2} Y^l_m(x) Y^j_n(x) dx = \text{cste} \; \delta_{lj} \delta_{mn}`

    The value of the constant depends on the choice of normalization.

    It obeys the following property:

    .. math::

        Y^{l+1}_i(x) &= \text{cste}(l) \; & C_{ijk} Y^l_j(x) x_k

        \partial_k Y^{l+1}_i(x) &= \text{cste}(l) \; (l+1) & C_{ijk} Y^l_j(x)

    Where :math:`C` are the `wigner_3j`.

    .. note::

        This function match with this table of standard real spherical harmonics from Wikipedia_
        when ``normalize=True``, ``normalization='integral'`` and is called with the argument in the order ``y,z,x`` (instead of ``x,y,z``).

    .. _Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Parameters
    ----------
    l : int or list of int
        degree of the spherical harmonics.

    x : `torch.Tensor`
        tensor :math:`x` of shape ``(..., 3)``.

    normalize : bool
        whether to normalize the ``x`` to unit vectors that lie on the sphere before projecting onto the spherical harmonics

    normalization : {'integral', 'component', 'norm'}
        normalization of the output tensors --- note that this option is independent of ``normalize``, which controls the processing of the *input*, rather than the output.
        Valid options:
        * *component*: :math:`\|Y^l(x)\|^2 = 2l+1, x \in S^2`
        * *norm*: :math:`\|Y^l(x)\| = 1, x \in S^2`, ``component / sqrt(2l+1)``
        * *integral*: :math:`\int_{S^2} Y^l_m(x)^2 dx = 1`, ``component / sqrt(4pi)``

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``

        .. math:: Y^l(x)

    Examples
    --------

    >>> spherical_harmonics(0, torch.randn(2, 3), False, normalization='component')
    tensor([[1.],
            [1.]])

    See Also
    --------
    wigner_D
    wigner_3j

    """
    sh = SphericalHarmonics(l, normalize, normalization)
    return sh(x)


@torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([
            sh_0_0,
        ], dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3) * x * z
    sh_2_1 = math.sqrt(3) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - (1/2) * x2z2
    sh_2_3 = math.sqrt(3) * y * z
    sh_2_4 = (1/2)*math.sqrt(3) * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)

    sh_3_0 = (1/6)*math.sqrt(30) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5) * sh_2_0 * y
    sh_3_2 = (1/8)*math.sqrt(24) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = (1/8)*math.sqrt(24) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5) * sh_2_4 * y
    sh_3_6 = (1/6)*math.sqrt(30) * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], dim=-1)

    sh_4_0 = (1/4)*math.sqrt(14)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (1/8)*math.sqrt(7)*(2*sh_3_0*y + math.sqrt(6)*sh_3_1*z + math.sqrt(6)*sh_3_5*x)
    sh_4_2 = -1/8*math.sqrt(2)*sh_3_0*z + (1/2)*math.sqrt(3)*sh_3_1*y + (1/8)*math.sqrt(30)*sh_3_2*z + (1/8)*math.sqrt(30)*sh_3_4*x + (1/8)*math.sqrt(2)*sh_3_6*x
    sh_4_3 = -1/8*math.sqrt(6)*sh_3_1*z + (1/4)*math.sqrt(15)*sh_3_2*y + (1/4)*math.sqrt(10)*sh_3_3*x + (1/8)*math.sqrt(6)*sh_3_5*x
    sh_4_4 = -1/4*math.sqrt(6)*sh_3_2*x + sh_3_3*y - 1/4*math.sqrt(6)*sh_3_4*z
    sh_4_5 = -1/8*math.sqrt(6)*sh_3_1*x + (1/4)*math.sqrt(10)*sh_3_3*z + (1/4)*math.sqrt(15)*sh_3_4*y - 1/8*math.sqrt(6)*sh_3_5*z
    sh_4_6 = -1/8*math.sqrt(2)*sh_3_0*x - 1/8*math.sqrt(30)*sh_3_2*x + (1/8)*math.sqrt(30)*sh_3_4*z + (1/2)*math.sqrt(3)*sh_3_5*y - 1/8*math.sqrt(2)*sh_3_6*z
    sh_4_7 = (1/8)*math.sqrt(7)*(-math.sqrt(6)*sh_3_1*x + math.sqrt(6)*sh_3_5*z + 2*sh_3_6*y)
    sh_4_8 = (1/4)*math.sqrt(14)*(-sh_3_0*x + sh_3_6*z)
    if lmax == 4:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], dim=-1)

    sh_5_0 = (3/10)*math.sqrt(10)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (3/5)*sh_4_0*y + (3/5)*math.sqrt(2)*sh_4_1*z + (3/5)*math.sqrt(2)*sh_4_7*x
    sh_5_2 = -1/10*math.sqrt(2)*sh_4_0*z + (4/5)*sh_4_1*y + (1/5)*math.sqrt(14)*sh_4_2*z + (1/5)*math.sqrt(14)*sh_4_6*x + (1/10)*math.sqrt(2)*sh_4_8*x
    sh_5_3 = -1/10*math.sqrt(6)*sh_4_1*z + (1/5)*math.sqrt(21)*sh_4_2*y + (1/10)*math.sqrt(42)*sh_4_3*z + (1/10)*math.sqrt(42)*sh_4_5*x + (1/10)*math.sqrt(6)*sh_4_7*x
    sh_5_4 = -1/5*math.sqrt(3)*sh_4_2*z + (2/5)*math.sqrt(6)*sh_4_3*y + (1/5)*math.sqrt(15)*sh_4_4*x + (1/5)*math.sqrt(3)*sh_4_6*x
    sh_5_5 = -1/5*math.sqrt(10)*sh_4_3*x + sh_4_4*y - 1/5*math.sqrt(10)*sh_4_5*z
    sh_5_6 = -1/5*math.sqrt(3)*sh_4_2*x + (1/5)*math.sqrt(15)*sh_4_4*z + (2/5)*math.sqrt(6)*sh_4_5*y - 1/5*math.sqrt(3)*sh_4_6*z
    sh_5_7 = -1/10*math.sqrt(6)*sh_4_1*x - 1/10*math.sqrt(42)*sh_4_3*x + (1/10)*math.sqrt(42)*sh_4_5*z + (1/5)*math.sqrt(21)*sh_4_6*y - 1/10*math.sqrt(6)*sh_4_7*z
    sh_5_8 = -1/10*math.sqrt(2)*sh_4_0*x - 1/5*math.sqrt(14)*sh_4_2*x + (1/5)*math.sqrt(14)*sh_4_6*z + (4/5)*sh_4_7*y - 1/10*math.sqrt(2)*sh_4_8*z
    sh_5_9 = -3/5*math.sqrt(2)*sh_4_1*x + (3/5)*math.sqrt(2)*sh_4_7*z + (3/5)*sh_4_8*y
    sh_5_10 = (3/10)*math.sqrt(10)*(-sh_4_0*x + sh_4_8*z)
    if lmax == 5:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
        ], dim=-1)

    sh_6_0 = (1/6)*math.sqrt(33)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*math.sqrt(11)*sh_5_0*y + (1/12)*math.sqrt(110)*sh_5_1*z + (1/12)*math.sqrt(110)*sh_5_9*x
    sh_6_2 = -1/12*math.sqrt(2)*sh_5_0*z + (1/3)*math.sqrt(5)*sh_5_1*y + (1/12)*math.sqrt(2)*sh_5_10*x + (1/4)*math.sqrt(10)*sh_5_2*z + (1/4)*math.sqrt(10)*sh_5_8*x
    sh_6_3 = -1/12*math.sqrt(6)*sh_5_1*z + (1/2)*math.sqrt(3)*sh_5_2*y + (1/2)*math.sqrt(2)*sh_5_3*z + (1/2)*math.sqrt(2)*sh_5_7*x + (1/12)*math.sqrt(6)*sh_5_9*x
    sh_6_4 = -1/6*math.sqrt(3)*sh_5_2*z + (2/3)*math.sqrt(2)*sh_5_3*y + (1/6)*math.sqrt(14)*sh_5_4*z + (1/6)*math.sqrt(14)*sh_5_6*x + (1/6)*math.sqrt(3)*sh_5_8*x
    sh_6_5 = -1/6*math.sqrt(5)*sh_5_3*z + (1/6)*math.sqrt(35)*sh_5_4*y + (1/6)*math.sqrt(21)*sh_5_5*x + (1/6)*math.sqrt(5)*sh_5_7*x
    sh_6_6 = -1/6*math.sqrt(15)*sh_5_4*x + sh_5_5*y - 1/6*math.sqrt(15)*sh_5_6*z
    sh_6_7 = -1/6*math.sqrt(5)*sh_5_3*x + (1/6)*math.sqrt(21)*sh_5_5*z + (1/6)*math.sqrt(35)*sh_5_6*y - 1/6*math.sqrt(5)*sh_5_7*z
    sh_6_8 = -1/6*math.sqrt(3)*sh_5_2*x - 1/6*math.sqrt(14)*sh_5_4*x + (1/6)*math.sqrt(14)*sh_5_6*z + (2/3)*math.sqrt(2)*sh_5_7*y - 1/6*math.sqrt(3)*sh_5_8*z
    sh_6_9 = -1/12*math.sqrt(6)*sh_5_1*x - 1/2*math.sqrt(2)*sh_5_3*x + (1/2)*math.sqrt(2)*sh_5_7*z + (1/2)*math.sqrt(3)*sh_5_8*y - 1/12*math.sqrt(6)*sh_5_9*z
    sh_6_10 = -1/12*math.sqrt(2)*sh_5_0*x - 1/12*math.sqrt(2)*sh_5_10*z - 1/4*math.sqrt(10)*sh_5_2*x + (1/4)*math.sqrt(10)*sh_5_8*z + (1/3)*math.sqrt(5)*sh_5_9*y
    sh_6_11 = -1/12*math.sqrt(110)*sh_5_1*x + (1/6)*math.sqrt(11)*sh_5_10*y + (1/12)*math.sqrt(110)*sh_5_9*z
    sh_6_12 = (1/6)*math.sqrt(33)*(-sh_5_0*x + sh_5_10*z)
    if lmax == 6:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
        ], dim=-1)

    sh_7_0 = (1/14)*math.sqrt(182)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*math.sqrt(13)*sh_6_0*y + (1/7)*math.sqrt(39)*sh_6_1*z + (1/7)*math.sqrt(39)*sh_6_11*x
    sh_7_2 = -1/14*math.sqrt(2)*sh_6_0*z + (2/7)*math.sqrt(6)*sh_6_1*y + (1/7)*math.sqrt(33)*sh_6_10*x + (1/14)*math.sqrt(2)*sh_6_12*x + (1/7)*math.sqrt(33)*sh_6_2*z
    sh_7_3 = -1/14*math.sqrt(6)*sh_6_1*z + (1/14)*math.sqrt(6)*sh_6_11*x + (1/7)*math.sqrt(33)*sh_6_2*y + (1/14)*math.sqrt(110)*sh_6_3*z + (1/14)*math.sqrt(110)*sh_6_9*x
    sh_7_4 = (1/7)*math.sqrt(3)*sh_6_10*x - 1/7*math.sqrt(3)*sh_6_2*z + (2/7)*math.sqrt(10)*sh_6_3*y + (3/14)*math.sqrt(10)*sh_6_4*z + (3/14)*math.sqrt(10)*sh_6_8*x
    sh_7_5 = -1/7*math.sqrt(5)*sh_6_3*z + (3/7)*math.sqrt(5)*sh_6_4*y + (3/7)*math.sqrt(2)*sh_6_5*z + (3/7)*math.sqrt(2)*sh_6_7*x + (1/7)*math.sqrt(5)*sh_6_9*x
    sh_7_6 = -1/14*math.sqrt(30)*sh_6_4*z + (4/7)*math.sqrt(3)*sh_6_5*y + (2/7)*math.sqrt(7)*sh_6_6*x + (1/14)*math.sqrt(30)*sh_6_8*x
    sh_7_7 = -1/7*math.sqrt(21)*sh_6_5*x + sh_6_6*y - 1/7*math.sqrt(21)*sh_6_7*z
    sh_7_8 = -1/14*math.sqrt(30)*sh_6_4*x + (2/7)*math.sqrt(7)*sh_6_6*z + (4/7)*math.sqrt(3)*sh_6_7*y - 1/14*math.sqrt(30)*sh_6_8*z
    sh_7_9 = -1/7*math.sqrt(5)*sh_6_3*x - 3/7*math.sqrt(2)*sh_6_5*x + (3/7)*math.sqrt(2)*sh_6_7*z + (3/7)*math.sqrt(5)*sh_6_8*y - 1/7*math.sqrt(5)*sh_6_9*z
    sh_7_10 = -1/7*math.sqrt(3)*sh_6_10*z - 1/7*math.sqrt(3)*sh_6_2*x - 3/14*math.sqrt(10)*sh_6_4*x + (3/14)*math.sqrt(10)*sh_6_8*z + (2/7)*math.sqrt(10)*sh_6_9*y
    sh_7_11 = -1/14*math.sqrt(6)*sh_6_1*x + (1/7)*math.sqrt(33)*sh_6_10*y - 1/14*math.sqrt(6)*sh_6_11*z - 1/14*math.sqrt(110)*sh_6_3*x + (1/14)*math.sqrt(110)*sh_6_9*z
    sh_7_12 = -1/14*math.sqrt(2)*sh_6_0*x + (1/7)*math.sqrt(33)*sh_6_10*z + (2/7)*math.sqrt(6)*sh_6_11*y - 1/14*math.sqrt(2)*sh_6_12*z - 1/7*math.sqrt(33)*sh_6_2*x
    sh_7_13 = -1/7*math.sqrt(39)*sh_6_1*x + (1/7)*math.sqrt(39)*sh_6_11*z + (1/7)*math.sqrt(13)*sh_6_12*y
    sh_7_14 = (1/14)*math.sqrt(182)*(-sh_6_0*x + sh_6_12*z)
    if lmax == 7:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
        ], dim=-1)

    sh_8_0 = (1/4)*math.sqrt(15)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*math.sqrt(15)*sh_7_0*y + (1/16)*math.sqrt(210)*sh_7_1*z + (1/16)*math.sqrt(210)*sh_7_13*x
    sh_8_2 = -1/16*math.sqrt(2)*sh_7_0*z + (1/4)*math.sqrt(7)*sh_7_1*y + (1/16)*math.sqrt(182)*sh_7_12*x + (1/16)*math.sqrt(2)*sh_7_14*x + (1/16)*math.sqrt(182)*sh_7_2*z
    sh_8_3 = (1/1360)*math.sqrt(510)*(-math.sqrt(85)*sh_7_1*z + math.sqrt(2210)*sh_7_11*x + math.sqrt(85)*sh_7_13*x + math.sqrt(2210)*sh_7_2*y + math.sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/8)*math.sqrt(33)*sh_7_10*x + (1/8)*math.sqrt(3)*sh_7_12*x - 1/8*math.sqrt(3)*sh_7_2*z + (1/2)*math.sqrt(3)*sh_7_3*y + (1/8)*math.sqrt(33)*sh_7_4*z
    sh_8_5 = (1/816)*math.sqrt(510)*(math.sqrt(102)*sh_7_11*x - math.sqrt(102)*sh_7_3*z + math.sqrt(1122)*sh_7_4*y + math.sqrt(561)*sh_7_5*z + math.sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*math.sqrt(30)*sh_7_10*x - 1/16*math.sqrt(30)*sh_7_4*z + (1/4)*math.sqrt(15)*sh_7_5*y + (3/16)*math.sqrt(10)*sh_7_6*z + (3/16)*math.sqrt(10)*sh_7_8*x
    sh_8_7 = -1/16*math.sqrt(42)*sh_7_5*z + (3/8)*math.sqrt(7)*sh_7_6*y + (3/4)*sh_7_7*x + (1/16)*math.sqrt(42)*sh_7_9*x
    sh_8_8 = -1/4*math.sqrt(7)*sh_7_6*x + sh_7_7*y - 1/4*math.sqrt(7)*sh_7_8*z
    sh_8_9 = -1/16*math.sqrt(42)*sh_7_5*x + (3/4)*sh_7_7*z + (3/8)*math.sqrt(7)*sh_7_8*y - 1/16*math.sqrt(42)*sh_7_9*z
    sh_8_10 = -1/16*math.sqrt(30)*sh_7_10*z - 1/16*math.sqrt(30)*sh_7_4*x - 3/16*math.sqrt(10)*sh_7_6*x + (3/16)*math.sqrt(10)*sh_7_8*z + (1/4)*math.sqrt(15)*sh_7_9*y
    sh_8_11 = (1/816)*math.sqrt(510)*(math.sqrt(1122)*sh_7_10*y - math.sqrt(102)*sh_7_11*z - math.sqrt(102)*sh_7_3*x - math.sqrt(561)*sh_7_5*x + math.sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/8)*math.sqrt(33)*sh_7_10*z + (1/2)*math.sqrt(3)*sh_7_11*y - 1/8*math.sqrt(3)*sh_7_12*z - 1/8*math.sqrt(3)*sh_7_2*x - 1/8*math.sqrt(33)*sh_7_4*x
    sh_8_13 = (1/1360)*math.sqrt(510)*(-math.sqrt(85)*sh_7_1*x + math.sqrt(2210)*sh_7_11*z + math.sqrt(2210)*sh_7_12*y - math.sqrt(85)*sh_7_13*z - math.sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/16*math.sqrt(2)*sh_7_0*x + (1/16)*math.sqrt(182)*sh_7_12*z + (1/4)*math.sqrt(7)*sh_7_13*y - 1/16*math.sqrt(2)*sh_7_14*z - 1/16*math.sqrt(182)*sh_7_2*x
    sh_8_15 = -1/16*math.sqrt(210)*sh_7_1*x + (1/16)*math.sqrt(210)*sh_7_13*z + (1/8)*math.sqrt(15)*sh_7_14*y
    sh_8_16 = (1/4)*math.sqrt(15)*(-sh_7_0*x + sh_7_14*z)
    if lmax == 8:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
        ], dim=-1)

    sh_9_0 = (1/6)*math.sqrt(34)*(sh_8_0*z + sh_8_16*x)
    sh_9_1 = (1/9)*math.sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)
    sh_9_2 = -1/18*math.sqrt(2)*sh_8_0*z + (4/9)*math.sqrt(2)*sh_8_1*y + (2/9)*math.sqrt(15)*sh_8_14*x + (1/18)*math.sqrt(2)*sh_8_16*x + (2/9)*math.sqrt(15)*sh_8_2*z
    sh_9_3 = -1/18*math.sqrt(6)*sh_8_1*z + (1/18)*math.sqrt(210)*sh_8_13*x + (1/18)*math.sqrt(6)*sh_8_15*x + (1/3)*math.sqrt(5)*sh_8_2*y + (1/18)*math.sqrt(210)*sh_8_3*z
    sh_9_4 = (1/18)*math.sqrt(182)*sh_8_12*x + (1/9)*math.sqrt(3)*sh_8_14*x - 1/9*math.sqrt(3)*sh_8_2*z + (2/9)*math.sqrt(14)*sh_8_3*y + (1/18)*math.sqrt(182)*sh_8_4*z
    sh_9_5 = (1/9)*math.sqrt(39)*sh_8_11*x + (1/9)*math.sqrt(5)*sh_8_13*x - 1/9*math.sqrt(5)*sh_8_3*z + (1/9)*math.sqrt(65)*sh_8_4*y + (1/9)*math.sqrt(39)*sh_8_5*z
    sh_9_6 = (1/9)*math.sqrt(33)*sh_8_10*x + (1/18)*math.sqrt(30)*sh_8_12*x - 1/18*math.sqrt(30)*sh_8_4*z + (2/3)*math.sqrt(2)*sh_8_5*y + (1/9)*math.sqrt(33)*sh_8_6*z
    sh_9_7 = (1/18)*math.sqrt(42)*sh_8_11*x - 1/18*math.sqrt(42)*sh_8_5*z + (1/9)*math.sqrt(77)*sh_8_6*y + (1/18)*math.sqrt(110)*sh_8_7*z + (1/18)*math.sqrt(110)*sh_8_9*x
    sh_9_8 = (1/9)*math.sqrt(14)*sh_8_10*x - 1/9*math.sqrt(14)*sh_8_6*z + (4/9)*math.sqrt(5)*sh_8_7*y + (1/3)*math.sqrt(5)*sh_8_8*x
    sh_9_9 = -2/3*sh_8_7*x + sh_8_8*y - 2/3*sh_8_9*z
    sh_9_10 = -1/9*math.sqrt(14)*sh_8_10*z - 1/9*math.sqrt(14)*sh_8_6*x + (1/3)*math.sqrt(5)*sh_8_8*z + (4/9)*math.sqrt(5)*sh_8_9*y
    sh_9_11 = (1/9)*math.sqrt(77)*sh_8_10*y - 1/18*math.sqrt(42)*sh_8_11*z - 1/18*math.sqrt(42)*sh_8_5*x - 1/18*math.sqrt(110)*sh_8_7*x + (1/18)*math.sqrt(110)*sh_8_9*z
    sh_9_12 = (1/9)*math.sqrt(33)*sh_8_10*z + (2/3)*math.sqrt(2)*sh_8_11*y - 1/18*math.sqrt(30)*sh_8_12*z - 1/18*math.sqrt(30)*sh_8_4*x - 1/9*math.sqrt(33)*sh_8_6*x
    sh_9_13 = (1/9)*math.sqrt(39)*sh_8_11*z + (1/9)*math.sqrt(65)*sh_8_12*y - 1/9*math.sqrt(5)*sh_8_13*z - 1/9*math.sqrt(5)*sh_8_3*x - 1/9*math.sqrt(39)*sh_8_5*x
    sh_9_14 = (1/18)*math.sqrt(182)*sh_8_12*z + (2/9)*math.sqrt(14)*sh_8_13*y - 1/9*math.sqrt(3)*sh_8_14*z - 1/9*math.sqrt(3)*sh_8_2*x - 1/18*math.sqrt(182)*sh_8_4*x
    sh_9_15 = -1/18*math.sqrt(6)*sh_8_1*x + (1/18)*math.sqrt(210)*sh_8_13*z + (1/3)*math.sqrt(5)*sh_8_14*y - 1/18*math.sqrt(6)*sh_8_15*z - 1/18*math.sqrt(210)*sh_8_3*x
    sh_9_16 = -1/18*math.sqrt(2)*sh_8_0*x + (2/9)*math.sqrt(15)*sh_8_14*z + (4/9)*math.sqrt(2)*sh_8_15*y - 1/18*math.sqrt(2)*sh_8_16*z - 2/9*math.sqrt(15)*sh_8_2*x
    sh_9_17 = (1/9)*math.sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)
    sh_9_18 = (1/6)*math.sqrt(34)*(-sh_8_0*x + sh_8_16*z)
    if lmax == 9:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
        ], dim=-1)

    sh_10_0 = (1/10)*math.sqrt(95)*(sh_9_0*z + sh_9_18*x)
    sh_10_1 = (1/10)*math.sqrt(19)*sh_9_0*y + (3/20)*math.sqrt(38)*sh_9_1*z + (3/20)*math.sqrt(38)*sh_9_17*x
    sh_10_2 = -1/20*math.sqrt(2)*sh_9_0*z + (3/5)*sh_9_1*y + (3/20)*math.sqrt(34)*sh_9_16*x + (1/20)*math.sqrt(2)*sh_9_18*x + (3/20)*math.sqrt(34)*sh_9_2*z
    sh_10_3 = -1/20*math.sqrt(6)*sh_9_1*z + (1/5)*math.sqrt(17)*sh_9_15*x + (1/20)*math.sqrt(6)*sh_9_17*x + (1/10)*math.sqrt(51)*sh_9_2*y + (1/5)*math.sqrt(17)*sh_9_3*z
    sh_10_4 = (1/5)*math.sqrt(15)*sh_9_14*x + (1/10)*math.sqrt(3)*sh_9_16*x - 1/10*math.sqrt(3)*sh_9_2*z + (4/5)*sh_9_3*y + (1/5)*math.sqrt(15)*sh_9_4*z
    sh_10_5 = (1/20)*math.sqrt(210)*sh_9_13*x + (1/10)*math.sqrt(5)*sh_9_15*x - 1/10*math.sqrt(5)*sh_9_3*z + (1/2)*math.sqrt(3)*sh_9_4*y + (1/20)*math.sqrt(210)*sh_9_5*z
    sh_10_6 = (1/20)*math.sqrt(182)*sh_9_12*x + (1/20)*math.sqrt(30)*sh_9_14*x - 1/20*math.sqrt(30)*sh_9_4*z + (1/5)*math.sqrt(21)*sh_9_5*y + (1/20)*math.sqrt(182)*sh_9_6*z
    sh_10_7 = (1/10)*math.sqrt(39)*sh_9_11*x + (1/20)*math.sqrt(42)*sh_9_13*x - 1/20*math.sqrt(42)*sh_9_5*z + (1/10)*math.sqrt(91)*sh_9_6*y + (1/10)*math.sqrt(39)*sh_9_7*z
    sh_10_8 = (1/10)*math.sqrt(33)*sh_9_10*x + (1/10)*math.sqrt(14)*sh_9_12*x - 1/10*math.sqrt(14)*sh_9_6*z + (2/5)*math.sqrt(6)*sh_9_7*y + (1/10)*math.sqrt(33)*sh_9_8*z
    sh_10_9 = (3/10)*math.sqrt(2)*sh_9_11*x - 3/10*math.sqrt(2)*sh_9_7*z + (3/10)*math.sqrt(11)*sh_9_8*y + (1/10)*math.sqrt(55)*sh_9_9*x
    sh_10_10 = -3/10*math.sqrt(5)*sh_9_10*z - 3/10*math.sqrt(5)*sh_9_8*x + sh_9_9*y
    sh_10_11 = (3/10)*math.sqrt(11)*sh_9_10*y - 3/10*math.sqrt(2)*sh_9_11*z - 3/10*math.sqrt(2)*sh_9_7*x + (1/10)*math.sqrt(55)*sh_9_9*z
    sh_10_12 = (1/10)*math.sqrt(33)*sh_9_10*z + (2/5)*math.sqrt(6)*sh_9_11*y - 1/10*math.sqrt(14)*sh_9_12*z - 1/10*math.sqrt(14)*sh_9_6*x - 1/10*math.sqrt(33)*sh_9_8*x
    sh_10_13 = (1/10)*math.sqrt(39)*sh_9_11*z + (1/10)*math.sqrt(91)*sh_9_12*y - 1/20*math.sqrt(42)*sh_9_13*z - 1/20*math.sqrt(42)*sh_9_5*x - 1/10*math.sqrt(39)*sh_9_7*x
    sh_10_14 = (1/20)*math.sqrt(182)*sh_9_12*z + (1/5)*math.sqrt(21)*sh_9_13*y - 1/20*math.sqrt(30)*sh_9_14*z - 1/20*math.sqrt(30)*sh_9_4*x - 1/20*math.sqrt(182)*sh_9_6*x
    sh_10_15 = (1/20)*math.sqrt(210)*sh_9_13*z + (1/2)*math.sqrt(3)*sh_9_14*y - 1/10*math.sqrt(5)*sh_9_15*z - 1/10*math.sqrt(5)*sh_9_3*x - 1/20*math.sqrt(210)*sh_9_5*x
    sh_10_16 = (1/5)*math.sqrt(15)*sh_9_14*z + (4/5)*sh_9_15*y - 1/10*math.sqrt(3)*sh_9_16*z - 1/10*math.sqrt(3)*sh_9_2*x - 1/5*math.sqrt(15)*sh_9_4*x
    sh_10_17 = -1/20*math.sqrt(6)*sh_9_1*x + (1/5)*math.sqrt(17)*sh_9_15*z + (1/10)*math.sqrt(51)*sh_9_16*y - 1/20*math.sqrt(6)*sh_9_17*z - 1/5*math.sqrt(17)*sh_9_3*x
    sh_10_18 = -1/20*math.sqrt(2)*sh_9_0*x + (3/20)*math.sqrt(34)*sh_9_16*z + (3/5)*sh_9_17*y - 1/20*math.sqrt(2)*sh_9_18*z - 3/20*math.sqrt(34)*sh_9_2*x
    sh_10_19 = -3/20*math.sqrt(38)*sh_9_1*x + (3/20)*math.sqrt(38)*sh_9_17*z + (1/10)*math.sqrt(19)*sh_9_18*y
    sh_10_20 = (1/10)*math.sqrt(95)*(-sh_9_0*x + sh_9_18*z)
    if lmax == 10:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
        ], dim=-1)

    sh_11_0 = (1/22)*math.sqrt(462)*(sh_10_0*z + sh_10_20*x)
    sh_11_1 = (1/11)*math.sqrt(21)*sh_10_0*y + (1/11)*math.sqrt(105)*sh_10_1*z + (1/11)*math.sqrt(105)*sh_10_19*x
    sh_11_2 = -1/22*math.sqrt(2)*sh_10_0*z + (2/11)*math.sqrt(10)*sh_10_1*y + (1/11)*math.sqrt(95)*sh_10_18*x + (1/11)*math.sqrt(95)*sh_10_2*z + (1/22)*math.sqrt(2)*sh_10_20*x
    sh_11_3 = -1/22*math.sqrt(6)*sh_10_1*z + (3/22)*math.sqrt(38)*sh_10_17*x + (1/22)*math.sqrt(6)*sh_10_19*x + (1/11)*math.sqrt(57)*sh_10_2*y + (3/22)*math.sqrt(38)*sh_10_3*z
    sh_11_4 = (3/22)*math.sqrt(34)*sh_10_16*x + (1/11)*math.sqrt(3)*sh_10_18*x - 1/11*math.sqrt(3)*sh_10_2*z + (6/11)*math.sqrt(2)*sh_10_3*y + (3/22)*math.sqrt(34)*sh_10_4*z
    sh_11_5 = (2/11)*math.sqrt(17)*sh_10_15*x + (1/11)*math.sqrt(5)*sh_10_17*x - 1/11*math.sqrt(5)*sh_10_3*z + (1/11)*math.sqrt(85)*sh_10_4*y + (2/11)*math.sqrt(17)*sh_10_5*z
    sh_11_6 = (2/11)*math.sqrt(15)*sh_10_14*x + (1/22)*math.sqrt(30)*sh_10_16*x - 1/22*math.sqrt(30)*sh_10_4*z + (4/11)*math.sqrt(6)*sh_10_5*y + (2/11)*math.sqrt(15)*sh_10_6*z
    sh_11_7 = (1/22)*math.sqrt(210)*sh_10_13*x + (1/22)*math.sqrt(42)*sh_10_15*x - 1/22*math.sqrt(42)*sh_10_5*z + (1/11)*math.sqrt(105)*sh_10_6*y + (1/22)*math.sqrt(210)*sh_10_7*z
    sh_11_8 = (1/22)*math.sqrt(182)*sh_10_12*x + (1/11)*math.sqrt(14)*sh_10_14*x - 1/11*math.sqrt(14)*sh_10_6*z + (4/11)*math.sqrt(7)*sh_10_7*y + (1/22)*math.sqrt(182)*sh_10_8*z
    sh_11_9 = (1/19481)*math.sqrt(5313)*(math.sqrt(23023)*sh_10_11*x + math.sqrt(10626)*sh_10_13*x - math.sqrt(10626)*sh_10_7*z + math.sqrt(69069)*sh_10_8*y + math.sqrt(23023)*sh_10_9*z)
    sh_11_10 = (1/11)*math.sqrt(66)*sh_10_10*x + (3/22)*math.sqrt(10)*sh_10_12*x - 3/22*math.sqrt(10)*sh_10_8*z + (2/11)*math.sqrt(30)*sh_10_9*y
    sh_11_11 = sh_10_10*y - 1/11*math.sqrt(55)*sh_10_11*z - 1/11*math.sqrt(55)*sh_10_9*x
    sh_11_12 = (1/11)*math.sqrt(66)*sh_10_10*z + (2/11)*math.sqrt(30)*sh_10_11*y - 3/22*math.sqrt(10)*sh_10_12*z - 3/22*math.sqrt(10)*sh_10_8*x
    sh_11_13 = (1/19481)*math.sqrt(5313)*(math.sqrt(23023)*sh_10_11*z + math.sqrt(69069)*sh_10_12*y - math.sqrt(10626)*sh_10_13*z - math.sqrt(10626)*sh_10_7*x - math.sqrt(23023)*sh_10_9*x)
    sh_11_14 = (1/22)*math.sqrt(182)*sh_10_12*z + (4/11)*math.sqrt(7)*sh_10_13*y - 1/11*math.sqrt(14)*sh_10_14*z - 1/11*math.sqrt(14)*sh_10_6*x - 1/22*math.sqrt(182)*sh_10_8*x
    sh_11_15 = (1/22)*math.sqrt(210)*sh_10_13*z + (1/11)*math.sqrt(105)*sh_10_14*y - 1/22*math.sqrt(42)*sh_10_15*z - 1/22*math.sqrt(42)*sh_10_5*x - 1/22*math.sqrt(210)*sh_10_7*x
    sh_11_16 = (2/11)*math.sqrt(15)*sh_10_14*z + (4/11)*math.sqrt(6)*sh_10_15*y - 1/22*math.sqrt(30)*sh_10_16*z - 1/22*math.sqrt(30)*sh_10_4*x - 2/11*math.sqrt(15)*sh_10_6*x
    sh_11_17 = (2/11)*math.sqrt(17)*sh_10_15*z + (1/11)*math.sqrt(85)*sh_10_16*y - 1/11*math.sqrt(5)*sh_10_17*z - 1/11*math.sqrt(5)*sh_10_3*x - 2/11*math.sqrt(17)*sh_10_5*x
    sh_11_18 = (3/22)*math.sqrt(34)*sh_10_16*z + (6/11)*math.sqrt(2)*sh_10_17*y - 1/11*math.sqrt(3)*sh_10_18*z - 1/11*math.sqrt(3)*sh_10_2*x - 3/22*math.sqrt(34)*sh_10_4*x
    sh_11_19 = -1/22*math.sqrt(6)*sh_10_1*x + (3/22)*math.sqrt(38)*sh_10_17*z + (1/11)*math.sqrt(57)*sh_10_18*y - 1/22*math.sqrt(6)*sh_10_19*z - 3/22*math.sqrt(38)*sh_10_3*x
    sh_11_20 = -1/22*math.sqrt(2)*sh_10_0*x + (1/11)*math.sqrt(95)*sh_10_18*z + (2/11)*math.sqrt(10)*sh_10_19*y - 1/11*math.sqrt(95)*sh_10_2*x - 1/22*math.sqrt(2)*sh_10_20*z
    sh_11_21 = -1/11*math.sqrt(105)*sh_10_1*x + (1/11)*math.sqrt(105)*sh_10_19*z + (1/11)*math.sqrt(21)*sh_10_20*y
    sh_11_22 = (1/22)*math.sqrt(462)*(-sh_10_0*x + sh_10_20*z)
    if lmax == 11:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
            sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
        ], dim=-1)

    sh_12_0 = (1/12)*math.sqrt(138)*(sh_11_0*z + sh_11_22*x)
    sh_12_1 = (1/12)*math.sqrt(23)*sh_11_0*y + (1/24)*math.sqrt(506)*sh_11_1*z + (1/24)*math.sqrt(506)*sh_11_21*x
    sh_12_2 = -1/24*math.sqrt(2)*sh_11_0*z + (1/6)*math.sqrt(11)*sh_11_1*y + (1/24)*math.sqrt(462)*sh_11_2*z + (1/24)*math.sqrt(462)*sh_11_20*x + (1/24)*math.sqrt(2)*sh_11_22*x
    sh_12_3 = -1/24*math.sqrt(6)*sh_11_1*z + (1/12)*math.sqrt(105)*sh_11_19*x + (1/4)*math.sqrt(7)*sh_11_2*y + (1/24)*math.sqrt(6)*sh_11_21*x + (1/12)*math.sqrt(105)*sh_11_3*z
    sh_12_4 = (1/12)*math.sqrt(95)*sh_11_18*x - 1/12*math.sqrt(3)*sh_11_2*z + (1/12)*math.sqrt(3)*sh_11_20*x + (1/3)*math.sqrt(5)*sh_11_3*y + (1/12)*math.sqrt(95)*sh_11_4*z
    sh_12_5 = (1/8)*math.sqrt(38)*sh_11_17*x + (1/12)*math.sqrt(5)*sh_11_19*x - 1/12*math.sqrt(5)*sh_11_3*z + (1/12)*math.sqrt(95)*sh_11_4*y + (1/8)*math.sqrt(38)*sh_11_5*z
    sh_12_6 = (1/8)*math.sqrt(34)*sh_11_16*x + (1/24)*math.sqrt(30)*sh_11_18*x - 1/24*math.sqrt(30)*sh_11_4*z + (1/2)*math.sqrt(3)*sh_11_5*y + (1/8)*math.sqrt(34)*sh_11_6*z
    sh_12_7 = (1/6)*math.sqrt(17)*sh_11_15*x + (1/24)*math.sqrt(42)*sh_11_17*x - 1/24*math.sqrt(42)*sh_11_5*z + (1/12)*math.sqrt(119)*sh_11_6*y + (1/6)*math.sqrt(17)*sh_11_7*z
    sh_12_8 = (1/6)*math.sqrt(15)*sh_11_14*x + (1/12)*math.sqrt(14)*sh_11_16*x - 1/12*math.sqrt(14)*sh_11_6*z + (2/3)*math.sqrt(2)*sh_11_7*y + (1/6)*math.sqrt(15)*sh_11_8*z
    sh_12_9 = (1/24)*math.sqrt(210)*sh_11_13*x + (1/4)*math.sqrt(2)*sh_11_15*x - 1/4*math.sqrt(2)*sh_11_7*z + (1/4)*math.sqrt(15)*sh_11_8*y + (1/24)*math.sqrt(210)*sh_11_9*z
    sh_12_10 = (1/24)*math.sqrt(182)*sh_11_10*z + (1/24)*math.sqrt(182)*sh_11_12*x + (1/8)*math.sqrt(10)*sh_11_14*x - 1/8*math.sqrt(10)*sh_11_8*z + (1/6)*math.sqrt(35)*sh_11_9*y
    sh_12_11 = (1/12)*math.sqrt(143)*sh_11_10*y + (1/12)*math.sqrt(78)*sh_11_11*x + (1/24)*math.sqrt(110)*sh_11_13*x - 1/24*math.sqrt(110)*sh_11_9*z
    sh_12_12 = -1/12*math.sqrt(66)*sh_11_10*x + sh_11_11*y - 1/12*math.sqrt(66)*sh_11_12*z
    sh_12_13 = (1/12)*math.sqrt(78)*sh_11_11*z + (1/12)*math.sqrt(143)*sh_11_12*y - 1/24*math.sqrt(110)*sh_11_13*z - 1/24*math.sqrt(110)*sh_11_9*x
    sh_12_14 = -1/24*math.sqrt(182)*sh_11_10*x + (1/24)*math.sqrt(182)*sh_11_12*z + (1/6)*math.sqrt(35)*sh_11_13*y - 1/8*math.sqrt(10)*sh_11_14*z - 1/8*math.sqrt(10)*sh_11_8*x
    sh_12_15 = (1/24)*math.sqrt(210)*sh_11_13*z + (1/4)*math.sqrt(15)*sh_11_14*y - 1/4*math.sqrt(2)*sh_11_15*z - 1/4*math.sqrt(2)*sh_11_7*x - 1/24*math.sqrt(210)*sh_11_9*x
    sh_12_16 = (1/6)*math.sqrt(15)*sh_11_14*z + (2/3)*math.sqrt(2)*sh_11_15*y - 1/12*math.sqrt(14)*sh_11_16*z - 1/12*math.sqrt(14)*sh_11_6*x - 1/6*math.sqrt(15)*sh_11_8*x
    sh_12_17 = (1/6)*math.sqrt(17)*sh_11_15*z + (1/12)*math.sqrt(119)*sh_11_16*y - 1/24*math.sqrt(42)*sh_11_17*z - 1/24*math.sqrt(42)*sh_11_5*x - 1/6*math.sqrt(17)*sh_11_7*x
    sh_12_18 = (1/8)*math.sqrt(34)*sh_11_16*z + (1/2)*math.sqrt(3)*sh_11_17*y - 1/24*math.sqrt(30)*sh_11_18*z - 1/24*math.sqrt(30)*sh_11_4*x - 1/8*math.sqrt(34)*sh_11_6*x
    sh_12_19 = (1/8)*math.sqrt(38)*sh_11_17*z + (1/12)*math.sqrt(95)*sh_11_18*y - 1/12*math.sqrt(5)*sh_11_19*z - 1/12*math.sqrt(5)*sh_11_3*x - 1/8*math.sqrt(38)*sh_11_5*x
    sh_12_20 = (1/12)*math.sqrt(95)*sh_11_18*z + (1/3)*math.sqrt(5)*sh_11_19*y - 1/12*math.sqrt(3)*sh_11_2*x - 1/12*math.sqrt(3)*sh_11_20*z - 1/12*math.sqrt(95)*sh_11_4*x
    sh_12_21 = -1/24*math.sqrt(6)*sh_11_1*x + (1/12)*math.sqrt(105)*sh_11_19*z + (1/4)*math.sqrt(7)*sh_11_20*y - 1/24*math.sqrt(6)*sh_11_21*z - 1/12*math.sqrt(105)*sh_11_3*x
    sh_12_22 = -1/24*math.sqrt(2)*sh_11_0*x - 1/24*math.sqrt(462)*sh_11_2*x + (1/24)*math.sqrt(462)*sh_11_20*z + (1/6)*math.sqrt(11)*sh_11_21*y - 1/24*math.sqrt(2)*sh_11_22*z
    sh_12_23 = -1/24*math.sqrt(506)*sh_11_1*x + (1/24)*math.sqrt(506)*sh_11_21*z + (1/12)*math.sqrt(23)*sh_11_22*y
    sh_12_24 = (1/12)*math.sqrt(138)*(-sh_11_0*x + sh_11_22*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
        sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
        sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24
    ], dim=-1)


def _generate_spherical_harmonics(lmax, device=None):  # pragma: no cover
    r"""code used to generate the code above

    based on `wigner_3j`
    """
    torch.set_default_dtype(torch.float64)

    def to_frac(x: float):
        from fractions import Fraction
        s = 1 if x >= 0 else -1
        x = x**2
        x = Fraction(x).limit_denominator()
        x = s * sympy.sqrt(x)
        x = sympy.simplify(x)
        return x

    print("sh_0_0 = torch.ones_like(x)")
    print("if lmax == 0:")
    print("    return torch.stack([")
    print("        sh_0_0,")
    print("    ], dim=-1)")
    print()

    x_var, y_var, z_var = sympy.symbols('x y z')
    polynomials = [x_var, y_var, z_var]
    poly_evalz = [0, 1, 0]

    def sub_z1(p, names, polynormz):
        p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
        for n, c in zip(names, polynormz):
            p = p.subs(n, c)
        return p

    for l in range(1, lmax+1):
        sh_variables = sympy.symbols(" ".join(f'sh_{l}_{m}' for m in range(2 * l + 1)))

        for n, p in zip(sh_variables, polynomials):
            print(f"{n} = {pycode(p)}")

        print(f"if lmax == {l}:")
        u = ",\n        ".join(", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(l + 1))
        print(f"    return torch.stack([\n        {u}\n    ], dim=-1)")
        print()

        if l == lmax:
            break

        polynomials = [
            sum(
                to_frac(c.item()) * v * sh
                for cj, v in zip(cij, [x_var, y_var, z_var])
                for c, sh in zip(cj, sh_variables)
            )
            for cij in o3.wigner_3j(l+1, 1, l, device=device)
        ]

        poly_evalz = [
            sub_z1(p, sh_variables, poly_evalz)
            for p in polynomials
        ]
        norm = sympy.sqrt(sum(p**2 for p in poly_evalz))
        polynomials = [p / norm for p in polynomials]
        poly_evalz = [p / norm for p in poly_evalz]

        polynomials = [
            sympy.simplify(p, full=True)
            for p in polynomials
        ]
