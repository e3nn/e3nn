# pylint: disable=not-callable, no-member, invalid-name, line-too-long, arguments-differ
"""
Fourier Transform : sphere (grid) <--> spherical tensor (Rs=[(1, l) for l in range(lmax + 1)])

We use the Fast Fourier Transform for specific
"""
import math

import lie_learn.spaces.S3 as S3
import torch
from e3nn import rsh, o3
from e3nn.util.default_dtype import torch_default_dtype


def s2_grid(res_beta, res_alpha):
    """
    grid on the sphere
    """
    i = torch.arange(res_beta).to(dtype=torch.get_default_dtype())
    betas = (i + 0.5) / res_beta * math.pi

    i = torch.arange(res_alpha).to(dtype=torch.get_default_dtype())
    alphas = i / res_alpha * 2 * math.pi
    return betas, alphas


def spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, _version=1):
    """
    computes the spherical harmonics on the grid on the sphere
    """
    with torch_default_dtype(torch.float64):
        betas, alphas = s2_grid(res_beta, res_alpha)
        sha = rsh.spherical_harmonics_alpha(lmax, alphas)  # [a, m]
        shb = rsh.spherical_harmonics_z(list(range(lmax + 1)), betas.cos(), betas.sin().abs())  # [b, l * m]
        return betas, alphas, shb, sha


def complete_lmax_res(lmax, res_beta, res_alpha):
    """
    try to use FFT
    i.e. 2 * lmax + 1 == res_alpha
    """
    if res_beta is None:
        res_beta = 2 * (lmax + 1)  # minimum req. to go on sphere and back

    if res_alpha is None:
        if lmax is not None:
            if res_beta is not None:
                res_alpha = max(2 * lmax + 1, res_beta - 1)
            else:
                res_alpha = 2 * lmax + 1  # minimum req. to go on sphere and back
        elif res_beta is not None:
            res_alpha = res_beta - 1

    if lmax is None:
        lmax = min(res_beta // 2 - 1, res_alpha // 2)  # maximum possible to go on sphere and back

    assert res_beta % 2 == 0
    assert lmax + 1 <= res_beta // 2

    return lmax, res_beta, res_alpha


def irfft(x, res):
    """
    :param x: tensor of shape [..., m]
    :return: tensor of shape [..., alpha]
    """
    assert res % 2 == 1
    *size, sm = x.shape
    x = x.reshape(-1, sm)
    x = torch.cat([
        x.new_zeros(x.shape[0], (res - sm) // 2),
        x,
        x.new_zeros(x.shape[0], (res - sm) // 2),
    ], dim=-1)
    assert x.shape[1] == res
    l = res // 2
    x = torch.stack([
        torch.cat([
            x[:, l:l + 1],
            x[:, l + 1:].div(math.sqrt(2))
        ], dim=1),
        torch.cat([
            torch.zeros_like(x[:, :1]),
            x[:, :l].flip(-1).div(-math.sqrt(2)),
        ], dim=1),
    ], dim=-1)
    x = torch.irfft(x, 1) * res
    return x.reshape(*size, res)


def rfft(x, l):
    """
    :param x: tensor of shape [..., alpha]
    :return: tensor of shape [..., m]
    """
    *size, res = x.shape
    x = x.reshape(-1, res)
    x = torch.rfft(x, 1)
    x = torch.cat([
        x[:, 1:l + 1, 1].flip(1).mul(-math.sqrt(2)),
        x[:, :1, 0],
        x[:, 1:l + 1, 0].mul(math.sqrt(2)),
    ], dim=1)
    return x.reshape(*size, 2 * l + 1)


class ToS2Grid(torch.nn.Module):
    """
    Transform spherical tensor into signal on the sphere

    The inverse transformation of FromS2Grid
    """

    def __init__(self, lmax=None, res=None, normalization='component'):
        """
        :param lmax: lmax of the input signal
        :param res: resolution of the output as a tuple (beta resolution, alpha resolution)
        :param normalization: either 'norm' or 'component'
        """
        super().__init__()

        assert normalization in ['norm', 'component', 'none'], "normalization needs to be 'norm', 'component' or 'none'"

        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = complete_lmax_res(lmax, *res)

        betas, alphas, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha)

        with torch_default_dtype(torch.float64):
            # normalize such that all l has the same variance on the sphere
            if normalization == 'component':
                n = math.sqrt(4 * math.pi) * torch.tensor([
                    1 / math.sqrt(2 * l + 1)
                    for l in range(lmax + 1)
                ]) / math.sqrt(lmax + 1)
            if normalization == 'norm':
                n = math.sqrt(4 * math.pi) * torch.ones(lmax + 1) / math.sqrt(lmax + 1)
            if normalization == 'none':
                n = torch.ones(lmax + 1)
            m = rsh.spherical_harmonics_expand_matrix(range(lmax + 1))  # [l, m, i]
        shb = torch.einsum('lmj,bj,lmi,l->mbi', m, shb, m, n)  # [m, b, i]

        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)
        self.to(torch.get_default_dtype())

    @property
    def grid(self):
        """
        grid of positions
        """
        beta, alpha = torch.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        """
        :param x: tensor [..., i=l * m]
        :return: tensor [..., beta, alpha]
        """
        size = x.shape[:-1]
        lmax = round(x.shape[-1] ** 0.5) - 1
        x = x.reshape(-1, (lmax + 1) ** 2)

        x = torch.einsum('mbi,zi->zbm', self.shb, x)  # [batch, beta, m]

        sa, sm = self.sha.shape
        if sa >= sm and sa % 2 == 1:
            x = irfft(x, sa)
        else:
            x = torch.einsum('am,zbm->zba', self.sha, x)
        return x.reshape(*size, *x.shape[1:])


class FromS2Grid(torch.nn.Module):
    """
    Transform signal on the sphere into spherical tensor

    The inverse transformation of ToS2Grid
    """

    def __init__(self, res=None, lmax=None, normalization='component', lmax_in=None):
        """
        :param res: resolution of the input as a tuple (beta resolution, alpha resolution)
        :param lmax: maximum l of the output
        :param normalization: either 'norm' or 'component'
        :param lmax_in: maximum l of the input of ToS2Grid in order to be the inverse
        """
        super().__init__()

        assert normalization in ['norm', 'component', 'none'], "normalization needs to be 'norm', 'component' or 'none'"

        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = complete_lmax_res(lmax, *res)

        if lmax_in is None:
            lmax_in = lmax

        betas, alphas, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha)

        with torch_default_dtype(torch.float64):
            # normalize such that it is the inverse of ToS2Grid
            if normalization == 'component':
                n = math.sqrt(4 * math.pi) * torch.tensor([
                    math.sqrt(2 * l + 1)
                    for l in range(lmax + 1)
                ]) * math.sqrt(lmax_in + 1)
            if normalization == 'norm':
                n = math.sqrt(4 * math.pi) * torch.ones(lmax + 1) * math.sqrt(lmax_in + 1)
            if normalization == 'none':
                n = 4 * math.pi * torch.ones(lmax + 1)
            m = rsh.spherical_harmonics_expand_matrix(range(lmax + 1))  # [l, m, i]
            assert res_beta % 2 == 0
            qw = torch.tensor(S3.quadrature_weights(res_beta // 2)) * res_beta**2 / res_alpha  # [b]
        shb = torch.einsum('lmj,bj,lmi,l,b->mbi', m, shb, m, n, qw)  # [m, b, i]

        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)
        self.to(torch.get_default_dtype())

    @property
    def grid(self):
        """
        grid of positions
        """
        beta, alpha = torch.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        """
        :param x: tensor [..., beta, alpha]
        :return: tensor [..., i=l * m]
        """
        size = x.shape[:-2]
        res_beta, res_alpha = x.shape[-2:]
        x = x.reshape(-1, res_beta, res_alpha)

        sa, sm = self.sha.shape
        if sm <= sa and sa % 2 == 1:
            x = rfft(x, sm // 2)
        else:
            x = torch.einsum('am,zba->zbm', self.sha, x)
        x = torch.einsum('mbi,zbm->zi', self.shb, x)
        return x.reshape(*size, x.shape[1])
