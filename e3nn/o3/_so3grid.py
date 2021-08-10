import torch
from e3nn.util.jit import compile_mode

from ._wigner import wigner_D
from ._s2grid import _quadrature_weights, s2_grid


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([(2 * l + 1)**0.5 * wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


@compile_mode('script')
class SO3Grid(torch.nn.Module):  # pylint: disable=abstract-method
    r'''Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax : int
        irreps representation ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}

    aspect_ratio : float
        default value (2) should be optimal
    '''
    def __init__(self, lmax, resolution, *, normalization='component', aspect_ratio=2):
        super().__init__()

        assert normalization == "component"

        nb = 2 * resolution
        na = round(2 * aspect_ratio * resolution)

        b, a = s2_grid(nb, na)
        self.register_buffer("D", flat_wigner(lmax, a[:, None, None], b[None, :, None], a[None, None, :]))
        qw = _quadrature_weights(nb // 2) * nb**2 / na**2
        self.register_buffer('qw', qw)

        self.register_buffer('alpha', a)
        self.register_buffer('beta', b)
        self.register_buffer('gamma', a)

        self.res_alpha = na
        self.res_beta = nb
        self.res_gamma = na

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.lmax})"

    def to_grid(self, features):
        r'''evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.irreps.dim)``

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
            tensor of shape ``(..., self.irreps.dim)``
        '''
        return torch.einsum("...abc,abci,b->...i", features, self.D, self.qw) * self.D.shape[-1]**0.5
