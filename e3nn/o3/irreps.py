import itertools
import collections
import warnings

import torch

from e3nn import o3
from e3nn.math import direct_sum, perm


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

    >>> Irrep("1o") + Irrep("2o")
    1x1o+1x2o
    """
    def __new__(cls, l, p=None):
        if isinstance(l, Irrep) and p is None:
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
        return super().__new__(cls, (l, p))

    def __repr__(self):
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

            if l == lmax:
                break

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Y axis, applied third.

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around X axis, applied second.

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Y axis, applied first.

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

    def D_from_matrix(self, R):
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

        Examples
        --------
        >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
        >>> m.long()
        tensor([[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]])
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
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

    def __mul__(self, ir):
        r"""generate the irreps from the product of two irreps

        Returns
        -------
        generator of `Irrep`
        """
        ir = Irrep(ir)
        p = self.p * ir.p
        lmin = abs(self.l - ir.l)
        lmax = self.l + ir.l
        return [Irrep(l, p) for l in range(lmin, lmax + 1)]

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, mul):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(mul, int)
        return Irreps([(mul, self)])

    def __add__(self, irreps):
        return Irreps(self) + Irreps(irreps)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> int:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


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
    def __new__(cls, irreps):
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
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
                out.append(_MulIr(mul, ir))
        else:
            for mul_ir in irreps:
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                elif len(mul_ir) == 3:
                    mul, l, p = mul_ir
                    ir = Irrep(l, p)
                    warnings.warn("prefer using [(mul, (l, p))] to distinguish multiplicity from irrep", DeprecationWarning, stacklevel=2)
                else:
                    mul = None
                    ir = None

                assert isinstance(mul, int) and mul >= 0
                assert ir is not None

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

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
        1x0e+1x1o+1x2e+1x3o
        """
        return Irreps([(1, (l, (-1)**l)) for l in range(lmax + 1)])

    def slices(self):
        r"""list of slice and ``mul_ir``

        Examples
        --------

        >>> Irreps('2x0e + 1e').slices()
        [(slice(0, 2, None), 2x0e), (slice(2, 5, None), 1x1e)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s += [(slice(i, i + mul_ir.dim), mul_ir)]
            i += mul_ir.dim
        return s

    def randn(self, *size, normalization='component', requires_grad=False, dtype=None, device=None):
        r"""random tensor

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
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)

        if normalization == 'norm':
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in self.slices():
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x

        assert False, "normalization needs to be 'norm' or 'component'"

    def __getitem__(self, i):
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (ir for _, ir in self)

    def count(self, ir) -> int:
        ir = Irrep(ir)
        return sum(mul for mul, ir in self if ir == ir)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps):
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
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
        2x1e+1x0e

        Same representations which are seperated from each other are not combined

        >>> Irreps("1e + 1e + 0e + 1e").simplify()
        2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def sort(self):
        r"""sort the representation

        Returns
        -------
        irreps : `Irreps`
        p : tuple of int
        inv : tuple of int

        Examples
        --------

        >>> Irreps("1e + 0e + 1e").sort().irreps
        1x0e+1x1e+1x1e

        >>> Irreps("2o + 1e + 0e + 1e").sort().p
        (3, 1, 0, 2)

        >>> Irreps("2o + 1e + 0e + 1e").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

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
        return "+".join(f"{mulir}" for mulir in self)

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

    def D_from_matrix(self, R):
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
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*o3.matrix_to_angles(R), k)
