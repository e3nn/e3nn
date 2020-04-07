# pylint: disable=not-callable, no-member, invalid-name, line-too-long, arguments-differ
"""
Fourier transform : sphere (grid) <--> spherical tensor (Rs=[(1, l) for l in range(lmax + 1)])
"""
import math

import lie_learn.spaces.S3 as S3
import torch

from e3nn import o3


def s2_grid(res_beta, res_alpha):
    """
    grid on the sphere
    """
    i = torch.arange(res_beta).to(dtype=torch.get_default_dtype())
    betas = (i + 0.5) / res_beta * math.pi

    i = torch.arange(res_alpha).to(dtype=torch.get_default_dtype())
    alphas = i / res_alpha * 2 * math.pi
    return betas, alphas


class ToS2Grid(torch.nn.Module):
    """
    Transform spherical tensor into signal on the sphere

    The inverse transformation of FromS2Grid
    """

    def __init__(self, lmax, res=None, normalization='component'):
        """
        :param lmax: lmax of the input signal
        :param normalization: either 'norm' or 'component'
        """
        super().__init__()

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

        if isinstance(res, int):
            res_beta, res_alpha = res, res
        elif res is None:
            res_beta = 2 * (lmax + 1)
            res_alpha = 2 * res_beta
        else:
            res_beta, res_alpha = res
        del res
        assert res_beta % 2 == 0
        assert res_beta >= 2 * (lmax + 1)

        betas, alphas = s2_grid(res_beta, res_alpha)
        sha = o3.spherical_harmonics_alpha_part(lmax, alphas)  # [m, a]
        shb = o3.spherical_harmonics_beta_part(lmax, betas.cos())  # [l, m, b]

        # normalize such that all l has the same variance on the sphere
        if normalization == 'component':
            n = math.sqrt(4 * math.pi) * torch.tensor([
                1 / math.sqrt(2 * l + 1)
                for l in range(lmax + 1)
            ]) / math.sqrt(lmax + 1)
        if normalization == 'norm':
            n = math.sqrt(4 * math.pi) * torch.ones(lmax + 1) / math.sqrt(lmax + 1)
        m = o3.spherical_harmonics_expand_matrix(lmax)  # [l, m, i]
        shb = torch.einsum('lmb,lmi,l->mbi', shb, m, n)  # [m, b, i]

        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)

    def forward(self, x):
        """
        :param x: tensor [..., i=l * m]
        :return: tensor [..., beta, alpha]
        """
        size = x.shape[:-1]
        lmax = round(x.shape[-1] ** 0.5) - 1
        x = x.reshape(-1, (lmax + 1) ** 2)
        out = torch.einsum('ma,zmb->zba', self.sha, torch.einsum('mbi,zi->zmb', self.shb, x))
        return out.view(*size, *out.shape[1:])


class FromS2Grid(torch.nn.Module):
    """
    Transform signal on the sphere into spherical tensor

    The inverse transformation of ToS2Grid
    """

    def __init__(self, res, lmax=None, normalization='component', lmax_in=None):
        """
        :param res: resolution of the input
        :param lmax: maximum l of the output
        :param normalization: either 'norm' or 'component'
        :param lmax_in: maximum l of the input of ToS2Grid in order to be the inverse
        """
        super().__init__()

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

        if isinstance(res, int):
            res_beta, res_alpha = res, res
        else:
            res_beta, res_alpha = res
        del res
        if lmax is None:
            lmax = res_beta // 2 - 1
        assert res_beta % 2 == 0
        assert lmax <= res_beta // 2 - 1
        if lmax_in is None:
            lmax_in = lmax

        betas, alphas = s2_grid(res_beta, res_alpha)
        sha = o3.spherical_harmonics_alpha_part(lmax, alphas)  # [m, a]
        shb = o3.spherical_harmonics_beta_part(lmax, betas.cos())  # [l, m, b]

        # normalize such that it is the inverse of ToS2Grid
        if normalization == 'component':
            n = math.sqrt(4 * math.pi) * torch.tensor([
                math.sqrt(2 * l + 1)
                for l in range(lmax + 1)
            ]) * math.sqrt(lmax_in + 1)
        if normalization == 'norm':
            n = math.sqrt(4 * math.pi) * torch.ones(lmax + 1) * math.sqrt(lmax_in + 1)
        m = o3.spherical_harmonics_expand_matrix(lmax)  # [l, m, i]
        qw = torch.tensor(S3.quadrature_weights(res_beta // 2)) * res_beta**2 / res_alpha  # [b]
        shb = torch.einsum('lmb,lmi,l,b->mbi', shb, m, n, qw)  # [m, b, i]

        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)

    def forward(self, x):
        """
        :param x: tensor [..., beta, alpha]
        :return: tensor [..., i=l * m]
        """
        size = x.shape[:-2]
        res_beta, res_alpha = x.shape[-2:]
        x = x.view(-1, res_beta, res_alpha)
        out = torch.einsum('mbi,zbm->zi', self.shb, torch.einsum('ma,zba->zbm', self.sha, x))
        return out.view(*size, out.shape[1])
