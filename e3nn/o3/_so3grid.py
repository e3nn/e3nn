import torch
from e3nn.util.jit import compile_mode

from ._irreps import Irreps
from ._wigner import wigner_D
from ._s2grid import _quadrature_weights, s2_grid


@compile_mode('script')
class SO3Grid(torch.nn.Module):  # pylint: disable=abstract-method
    r'''Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    irreps : `o3.Irreps`
        input representation of the form ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    resolution : int
        about ``resolution**3`` points are used to sample SO(3) (the higher the more accurate)

    normalization : {'norm', 'component'}
    '''
    def __init__(self, irreps: Irreps, resolution, normalization='component', aspect_ratio=2):
        super().__init__()

        irreps = Irreps(irreps).simplify()
        _, (_, p_val) = irreps[0]
        _, (lmax, _) = irreps[-1]
        assert all(mul == ir.dim for mul, ir in irreps)
        assert sorted(set(irreps.ls)) == list(range(lmax + 1))
        assert all(p == p_val for _, (l, p) in irreps), "the parity of the input is not well defined"
        self.irreps = irreps
        # the input transforms as : A_l ---> p_val * A_l
        # the signal transforms as : f(g) ---> p_val * f(g)

        assert normalization == "component"

        nb = 2 * resolution
        na = round(2 * aspect_ratio * resolution)

        b, a = s2_grid(nb, na)
        self.register_buffer(
            "D",
            torch.cat([
                (2 * l + 1)**0.5 * wigner_D(
                    l, a[:, None, None], b[None, :, None], a[None, None, :]
                ).transpose(-1, -2).reshape(na, nb, na, (2 * l + 1)**2)
                for l in range(lmax + 1)
            ], dim=-1)
        )
        qw = _quadrature_weights(nb // 2) * nb**2 / na**2
        self.register_buffer('qw', qw)

        self.register_buffer('alpha', a)
        self.register_buffer('beta', b)
        self.register_buffer('gamma', a)

        self.res_alpha = na
        self.res_beta = nb
        self.res_gamma = na

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in})"

    def to_grid(self, features):
        r'''evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor :math:`\{A^l\}_l` of shape ``(..., self.irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``
        '''
        return torch.einsum("...i,abci->...abc", features, self.D) / self.D.shape[-1]**0.5

    def from_grid(self, features):
        r'''evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``

        Returns
        -------
        `torch.Tensor`
            tensor :math:`\{A^l\}_l` of shape ``(..., self.irreps_in.dim)``
        '''
        return torch.einsum("...abc,abci,b->...i", features, self.D, self.qw) * self.D.shape[-1]**0.5
