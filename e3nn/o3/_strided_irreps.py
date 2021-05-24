import torch

from e3nn.math import direct_sum
from ._irreps import Irreps


class StridedIrreps(Irreps):
    def __new__(cls, irreps=None):
        self = super().__new__(cls, irreps=irreps)
        mul = self.mul
        # TODO: this could technically be anything with a common divisor... just use gcd?
        if any(this_mul > 0 and this_mul != mul for this_mul, _ in self):
            raise ValueError("The input irreps in `{self}` do not share a common multiplicity and thus cannot be represented in strided format.")
        return self

    # = New methods =

    @property
    def base_irreps(self) -> Irreps:
        return Irreps([(1, ir) for _, ir in self])

    @property
    def mul(self) -> int:
        if len(self) == 0:
            return 0
        else:
            return next(mul for mul, _ in self if mul > 0)

    @property
    def base_dim(self) -> int:
        return self.base_irreps.dim

    # = Overrides =

    def slices(self):
        raise NotImplementedError

    def randn(self, *size, normalization='component', requires_grad=False, dtype=None, device=None):
        assert size[-1] == -1

        if normalization == 'component':
            return torch.randn(size[:-1] + (self.dim,), requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == 'norm':
            raise NotImplementedError
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return StridedIrreps([mul*other, ir] for mul, ir in self)

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return StridedIrreps([mul*other, ir] for mul, ir in self)

    def simplify(self):
        # Can't simplify, since that would make the multiplicities inconsistant
        return self

    def __repr__(self):
        return super().__repr__() + " (strided)"

    # This implicity takes care of the other D_from_* methods
    def D_from_angles(self, alpha, beta, gamma, k=None):
        one_D = self.base_irreps.D_from_angles(alpha, beta, gamma, k=k)
        return direct_sum(*([one_D] * self.mul))
