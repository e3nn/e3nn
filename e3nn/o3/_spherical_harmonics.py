r"""Spherical Harmonics as polynomials of x, y, z
"""
from typing import Any, List, Union
import math

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from ._spherical_harmonics_generated import _get_spherical_harmonics


@compile_mode('trace')
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

        sh = _get_spherical_harmonics(self._lmax)(
            x[..., 0], x[..., 1], x[..., 2]
        )

        if not self._is_range_lmax:
            sh = torch.cat([
                sh[..., l*l:(l+1)*(l+1)]
                for l in self._ls_list
            ], dim=-1)

        # defualt normalization is component
        # correct for the other normalizations if asked for:
        if self.normalization == 'integral':
            sh.div_(math.sqrt(4 * math.pi))
        elif self.normalization == 'norm':
            sh.div_(torch.cat([
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


def _generate_spherical_harmonics(lmax, device=None):  # pragma: no cover
    r"""code used to generate the code above

    based on `wigner_3j`
    """
    import sympy
    from sympy.printing.pycode import pycode

    torch.set_default_dtype(torch.float64)

    def to_frac(x: float):
        from fractions import Fraction
        s = 1 if x >= 0 else -1
        x = x**2
        x = Fraction(x).limit_denominator()
        x = s * sympy.sqrt(x)
        x = sympy.simplify(x)
        return x

    outlines = [
        f"def _sph_lmax_{lmax}(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:",
    ]
    outlines.append("sh_0_0 = torch.ones_like(x)")

    x_var, y_var, z_var = sympy.symbols('x y z')
    polynomials = [sympy.sqrt(3) * x_var, sympy.sqrt(3) * y_var, sympy.sqrt(3) * z_var]

    def sub_z1(p, names, polynormz):
        p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
        for n, c in zip(names, polynormz):
            p = p.subs(n, c)
        return p

    poly_evalz = [
        sub_z1(p, [], [])
        for p in polynomials
    ]

    for l in range(1, lmax+1):
        sh_variables = sympy.symbols(" ".join(f'sh_{l}_{m}' for m in range(2 * l + 1)))

        for n, p in zip(sh_variables, polynomials):
            outlines.append(f"{n} = {pycode(p)}")

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
        # the sympy.sqrt is "component" normalization
        polynomials = [sympy.sqrt(2 * l + 3) * p / norm for p in polynomials]
        poly_evalz = [sympy.sqrt(2 * l + 3) * p / norm for p in poly_evalz]

        polynomials = [
            sympy.simplify(p, full=True)
            for p in polynomials
        ]

    u = ",\n        ".join(", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(lmax + 1))
    outlines.append(f"return torch.stack([\n        {u}\n    ], dim=-1)")

    # indent function body
    outlines = [outlines[0]] + ["    " + e for e in outlines[1:]]
    out = "\n".join(outlines)
    return out.replace(
        "math.sqrt",  # generated by sympy
        "sqrt",       # shorten the file and use alias
    )


def _generate_all_spherical_harmonics(lmax, device=None):  # pragma: no cover
    funcs = [
        _generate_spherical_harmonics(l, device=device)
        for l in range(lmax + 1)
    ]
    funcs = [
        "# flake8: noqa\n"
        "import torch\n"
        "import math\n"
        "import functools\n\n"
        "sqrt = math.sqrt\n"
    ] + funcs + [
        "_spherical_harmonics = {" + ", ".join(
            f"{l}: _sph_lmax_{l}"
            for l in range(lmax + 1)
        ) + "}",
        "@functools.lru_cache\n"
        "def _get_spherical_harmonics(lmax: int):\n"
        "    return torch.jit.script(_spherical_harmonics[lmax])\n"
    ]
    return "\n\n\n".join(funcs)
