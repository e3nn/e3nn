import itertools

import torch

from e3nn import o3
from e3nn.math import direct_sum


class Irrep(tuple):
    r"""Irreducible representation of :math:`O(3)`

    Attributes
    ----------
    l : int
        non negative integer, the degree of the representation, :math:`l = 0, 1, \dots`

    p : {1, -1}
        the parity of the representation

    dim : int
        the dimension of the representation

    Examples
    --------
    Create a representation :math:`l=0` of even parity.

    >>> Irrep(0, 1)
    0e

    Create a representation :math:`l=2` of odd parity.

    >>> Irrep(2, -1)
    2o

    Create a representation :math:`l=1` of the parity of the spherical harmonics.

    >>> Irrep("1y")
    1o

    >>> Irrep("2e") in Irrep("1o") * Irrep("1o")
    True
    """
    def __new__(self, l, p=None):
        if isinstance(l, Irrep):
            return l

        if isinstance(l, str) and p is None:
            name = l.strip()
            l = int(name[:-1])
            assert l >= 0
            p = {
                'e': 1,
                'o': -1,
                'y': (-1)**l,
            }[name[-1]]

        if isinstance(l, tuple) and p is None:
            l, p = l

        assert isinstance(l, int) and l >= 0, l
        assert p in [-1, 1], p
        return tuple.__new__(self, (l, p))

    def __repr__(self):
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls):
        r"""Iterator through all the irreps of :math:`O(3)`
        """
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Z axis, applied third.

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around Y axis, applied second.

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Z axis, applied first.

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the parity is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        See Also
        --------
        o3.wigner_D
        Irreps.D_from_angles
        """
        if k is None:
            k = torch.zeros_like(alpha)

        alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
        return o3.wigner_D(self.l, alpha, beta, gamma) * self.p**k[..., None, None]

    def D_from_quaternion(self, q, k=None):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*o3.quaternion_to_angles(q), k)

    def D_from_matrix(self, R, k=None):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*o3.matrix_to_angles(R), k)

    @property
    def l(self) -> int:
        return self[0]

    @property
    def p(self) -> int:
        return self[1]

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    def __mul__(self, other):
        r"""generate the irreps from the product of two irreps

        Returns
        -------
        generator of `Irrep`
        """
        assert isinstance(other, Irrep)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`

    Attributes
    ----------
    dim : int
        the total dimension of the representation

    num_irreps : int
        number of irreps. the sum of the multiplicities

    ls : list of int
        list of :math:`l` values

    lmax : int
        maximum :math:`l` value

    Examples
    --------
    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> Irreps([(100, (0, 1)), (50, (1, 1))])
    100x0e+50x1e

    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> Irreps("100x0e + 50x1e")
    100x0e+50x1e

    >>> Irreps("100x0e + 50x1e + 0x2e")
    100x0e+50x1e+0x2e

    >>> Irreps("100x0e + 50x1e + 0x2e").lmax
    1

    >>> Irrep("2e") in Irreps("0e + 2e")
    True
    """
    def __new__(self, irreps):
        if isinstance(irreps, Irreps):
            return tuple.__new__(self, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append((1, Irrep(irreps)))
        elif isinstance(irreps, str):
            for mul_ir in irreps.split('+'):
                if 'x' in mul_ir:
                    mul, ir = mul_ir.split('x')
                    mul = int(mul)
                    ir = Irrep(ir)
                else:
                    mul = 1
                    ir = Irrep(mul_ir)

                assert isinstance(mul, int) and mul >= 0
                out.append((mul, ir))
        else:
            for mul_ir in irreps:
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                elif len(mul_ir) == 3:
                    mul, l, p = mul_ir
                    ir = Irrep(l, p)
                else:
                    mul = None
                    ir = None

                assert isinstance(mul, int) and mul >= 0
                assert ir is not None

                out.append((mul, ir))
        return tuple.__new__(self, out)

    @staticmethod
    def spherical_harmonics(lmax):
        r"""representation of the spherical harmonics

        Parameters
        ----------
        lmax : int
            maximum :math:`l`

        Returns
        -------
        `Irreps`
            representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`

        Examples
        --------

        >>> Irreps.spherical_harmonics(3)
        0e+1o+2e+3o
        """
        return Irreps([(1, l, (-1)**l) for l in range(lmax + 1)])

    def randn(self, *size, normalization='component', dtype=None, device=None, requires_grad=False):
        """random tensor

        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``

        normalization : {'component', 'norm'}

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples
        --------

        >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 35, 5])

        >>> Irreps("2o").randn(2, -1, 3, normalization='norm').norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]

        if normalization == 'component':
            return torch.randn(*lsize, self.dim, *rsize, dtype=dtype, device=device, requires_grad=requires_grad)

        if normalization == 'norm':
            x = torch.zeros(*lsize, self.dim, *rsize, dtype=dtype, device=device, requires_grad=requires_grad)
            with torch.no_grad():
                start = 0
                for mul, ir in self:
                    r = torch.randn(*lsize, mul, ir.dim, *rsize)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
                    start += mul * ir.dim
            return x

        assert False, "normalization needs to be 'norm' or 'component'"

    def __getitem__(self, i):
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, x: object) -> bool:
        if isinstance(x, Irrep):
            return x in (ir for _, ir in self)
        return super().__contains__(x)

    def __add__(self, other):
        return Irreps(super().__add__(other))

    def __radd__(self, other):
        return Irreps(super().__radd__(other))

    def __mul__(self, other):
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        return Irreps(super().__rmul__(other))

    def simplify(self):
        """simplify the representation

        Returns
        -------
        `Irreps`

        Examples
        --------

        Note that simplify does not sort the representations.

        >>> Irreps("1e + 1e + 0e").simplify()
        2x1e+0e

        Same representations which are seperated from each other are not combined

        >>> Irreps("1e + 1e + 0e + 1e").simplify()
        2x1e+0e+1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self):
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        return max(self.ls)

    def __repr__(self):
        return "+".join("{}{}".format(f"{mul}x" if mul != 1 else "", ir) for mul, ir in self)

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    def D_from_quaternion(self, q, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*o3.quaternion_to_angles(q), k)

    def D_from_matrix(self, R, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*o3.matrix_to_angles(R), k)
