# pylint: disable=not-callable, no-member, invalid-name, line-too-long, arguments-differ
"""
The SOFT grid is a grid of the sphere
Fourier transform : sphere (SOFT res*res grid) <--> spherical tensor (Rs=[(1, l) for l in range(lmax + 1)])
"""
import math

import lie_learn.spaces.S3 as S3
import torch

from e3nn import o3


def soft_grid(res):
    """
    res x res SOFT grid on the sphere
    """
    i = torch.arange(res).to(dtype=torch.get_default_dtype())
    betas = (i + 0.5) / res * math.pi
    alphas = i / res * 2 * math.pi
    return betas, alphas


class ToSOFT(torch.nn.Module):
    """
    Transform spherical tensor into signal on the sphere
    """

    def __init__(self, lmax, res=None):
        """
        :param lmax: lmax of the input signal
        """
        super().__init__()

        if res is None:
            res = 2 * (lmax + 1)
        assert res % 2 == 0
        assert res >= 2 * (lmax + 1)

        betas, alphas = soft_grid(res)
        sha = o3.spherical_harmonics_alpha_part(lmax, alphas)  # [m, a]
        shb = o3.spherical_harmonics_beta_part(lmax, betas.cos())  # [l, m, b]

        # normalize such that all l has the same variance on the sphere
        n = math.sqrt(4 * math.pi) * torch.tensor([
            1 / math.sqrt(2 * l + 1)
            for l in range(lmax + 1)
        ]) / math.sqrt(lmax + 1)
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
        x = x.view(-1, (lmax + 1) ** 2)
        out = torch.einsum('ma,zmb->zba', self.sha, torch.einsum('mbi,zi->zmb', self.shb, x))
        return out.view(*size, *out.shape[1:])


class FromSOFT(torch.nn.Module):
    """
    Transform signal on the sphere into spherical tensor
    """

    def __init__(self, res, lmax=None):
        """
        :param res: resolution of the input signal
        """
        super().__init__()

        if lmax is None:
            lmax = res // 2 - 1
        assert res % 2 == 0
        assert lmax <= res // 2 - 1

        betas, alphas = soft_grid(res)
        sha = o3.spherical_harmonics_alpha_part(lmax, alphas)  # [m, a]
        shb = o3.spherical_harmonics_beta_part(lmax, betas.cos())  # [l, m, b]

        # normalize such that it is the inverse of ToSOFT
        n = math.sqrt(4 * math.pi) * torch.tensor([
            math.sqrt(2 * l + 1)
            for l in range(lmax + 1)
        ]) * math.sqrt(lmax + 1)
        m = o3.spherical_harmonics_expand_matrix(lmax)  # [l, m, i]
        qw = torch.tensor(S3.quadrature_weights(res // 2)) * res  # [b]
        shb = torch.einsum('lmb,lmi,l,b->mbi', shb, m, n, qw)  # [m, b, i]

        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)

    def forward(self, x):
        """
        :param x: tensor [..., beta, alpha]
        :return: tensor [..., i=l * m]
        """
        size = x.shape[:-2]
        res = x.shape[-1]
        x = x.view(-1, res, res)
        out = torch.einsum('mbi,zbm->zi', self.shb, torch.einsum('ma,zba->zbm', self.sha, x))
        return out.view(*size, out.shape[1])
