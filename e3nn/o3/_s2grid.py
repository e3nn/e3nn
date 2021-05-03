r"""Transformation between two representations of a signal on the sphere.

.. math:: f: S^2 \longrightarrow \mathbb{R}

is a signal on the sphere.

One representation that we like to call "spherical tensor" is

.. math:: f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)

it is made of :math:`(l_{\mathit{max}} + 1)^2` real numbers represented in the above formula by the familly of vectors :math:`F^l \in \mathbb{R}^{2l+1}`.

Another representation is the discretization around the sphere. For this representation we chose a particular grid of size :math:`(N, M)`

.. math::

    x_{ij} &= (\sin(\beta_i) \sin(\alpha_j), \cos(\beta_i), \sin(\beta_i) \cos(\alpha_j))

    \beta_i &= \pi (i + 0.5) / N

    \alpha_j &= 2 \pi j / M

In the code, :math:`N` is called ``res_beta`` and :math:`M` is ``res_alpha``.

The discrete representation is therefore

.. math:: \{ h_{ij} = f(x_{ij}) \}_{ij}
"""
import math

import torch
import torch.fft
from e3nn import o3
from e3nn.util import explicit_default_types


def _quadrature_weights(b, dtype=None, device=None):
    """
    function copied from ``lie_learn.spaces.S3``

    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].
    """
    k = torch.arange(b, device=device)
    w = torch.tensor([
        (
            (2. / b) * torch.sin(math.pi * (2. * j + 1.) / (4. * b)) * (
                (1. / (2 * k + 1)) * torch.sin((2 * j + 1) * (2 * k + 1) * math.pi / (4. * b))
            ).sum()
        )
        for j in torch.arange(2 * b, device=device)
    ], dtype=dtype, device=device)

    w /= 2. * ((2 * b) ** 2)
    return w


def s2_grid(res_beta, res_alpha, dtype=None, device=None):
    r"""grid on the sphere

    Parameters
    ----------
    res_beta : int
        :math:`N`

    res_alpha : int
        :math:`M`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensors. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensors. If ``None`` then set to the default device of the current context.

    Returns
    -------
    betas : `torch.Tensor`
        tensor of shape ``(res_beta)``

    alphas : `torch.Tensor`
        tensor of shape ``(res_alpha)``
    """
    dtype, device = explicit_default_types(dtype, device)

    i = torch.arange(res_beta, dtype=dtype, device=device)
    betas = (i + 0.5) / res_beta * math.pi

    i = torch.arange(res_alpha, dtype=dtype, device=device)
    alphas = i / res_alpha * 2 * math.pi
    return betas, alphas


def spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=None, device=None):
    r"""spherical harmonics evaluated on the grid on the sphere

    .. math::

        f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)

        f(\beta, \alpha) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot S^l(\alpha) P^l(\cos(\beta))

    Parameters
    ----------
    lmax : int
        :math:`l_{\mathit{max}}`

    res_beta : int
        :math:`N`

    res_alpha : int
        :math:`M`

    Returns
    -------
    betas : `torch.Tensor`
        tensor of shape ``(res_beta)``

    alphas : `torch.Tensor`
        tensor of shape ``(res_alpha)``

    shb : `torch.Tensor`
        tensor of shape ``(res_beta, (lmax + 1)**2)``

    sha : `torch.Tensor`
        tensor of shape ``(res_alpha, 2 lmax + 1)``
    """
    betas, alphas = s2_grid(res_beta, res_alpha, dtype=dtype, device=device)
    shb = o3.Legendre(list(range(lmax + 1)))(betas.cos(), betas.sin().abs())  # [b, l * m]
    sha = o3.spherical_harmonics_alpha(lmax, alphas)  # [a, m]
    return betas, alphas, shb, sha


def _complete_lmax_res(lmax, res_beta, res_alpha):
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


def _expand_matrix(ls, like=None, dtype=None, device=None):
    """
    convertion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    :return: tensor [l, m, l * m]
    """
    lmax = max(ls)
    if like is None:
        m = torch.zeros(len(ls),
                        2 * lmax + 1,
                        sum(2 * l + 1 for l in ls),
                        dtype=dtype,
                        device=device)
    else:
        m = like.new_zeros((len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls)),
                           dtype=dtype,
                           device=device)
    i = 0
    for j, l in enumerate(ls):
        m[j, lmax - l:lmax + l + 1, i:i + 2 * l + 1] = torch.eye(2 * l + 1, dtype=dtype, device=device)
        i += 2 * l + 1
    return m


def rfft(x, l):
    r"""Real fourier transform

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape ``(..., 2 l + 1)``

    res : int
        output resolution, has to be an odd number

    Returns
    -------
    `torch.Tensor`
        tensor of shape ``(..., res)``

    Examples
    --------

    >>> lmax = 8
    >>> res = 101
    >>> _betas, _alphas, _shb, sha = spherical_harmonics_s2_grid(lmax, res, res)
    >>> x = torch.randn(res)
    >>> (rfft(x, lmax) - x @ sha).abs().max().item() < 1e-4
    True
    """
    *size, res = x.shape
    x = x.reshape(-1, res)
    x = torch.fft.rfft(x, dim=1)
    x = torch.cat([
        x[:, 1:l + 1].imag.flip(1).mul(-math.sqrt(2)),
        x[:, :1].real,
        x[:, 1:l + 1].real.mul(math.sqrt(2)),
    ], dim=1)
    return x.reshape(*size, 2 * l + 1)


def irfft(x, res):
    r"""Inverse of the real fourier transform

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape ``(..., 2 l + 1)``

    res : int
        output resolution, has to be an odd number

    Returns
    -------
    `torch.Tensor`
        positions on the sphere, tensor of shape ``(..., res, 3)``

    Examples
    --------

    >>> lmax = 8
    >>> res = 101
    >>> _betas, _alphas, _shb, sha = spherical_harmonics_s2_grid(lmax, res, res)
    >>> x = torch.randn(2 * lmax + 1)
    >>> (irfft(x, res) - sha @ x).abs().max().item() < 1e-4
    True
    """
    assert res % 2 == 1
    *size, sm = x.shape
    x = x.reshape(-1, sm)
    x = torch.cat([
        x.new_zeros((x.shape[0], (res - sm) // 2)),
        x,
        x.new_zeros((x.shape[0], (res - sm) // 2)),
    ], dim=-1)
    assert x.shape[1] == res
    l = res // 2
    x = torch.complex(
        torch.cat([
            x[:, l:l + 1],
            x[:, l + 1:].div(math.sqrt(2))
        ], dim=1),
        torch.cat([
            torch.zeros_like(x[:, :1]),
            x[:, :l].flip(-1).div(-math.sqrt(2)),
        ], dim=1),
    )
    x = torch.fft.irfft(x, n=res, dim=1) * res
    return x.reshape(*size, res)


class ToS2Grid(torch.nn.Module):
    r"""Transform spherical tensor into signal on the sphere

    The inverse transformation of `FromS2Grid`

    Parameters
    ----------
    lmax : int
    res : int, tuple of int
    normalization : {'norm', 'component', 'integral'}

    Examples
    --------

    >>> m = ToS2Grid(6, (100, 101))
    >>> x = torch.randn(3, 49)
    >>> m(x).shape
    torch.Size([3, 100, 101])


    `ToS2Grid` and `FromS2Grid` are inverse of each other

    >>> m = ToS2Grid(6, (100, 101))
    >>> k = FromS2Grid((100, 101), 6)
    >>> x = torch.randn(3, 49)
    >>> y = k(m(x))
    >>> (x - y).abs().max().item() < 1e-4
    True

    Attributes
    ----------
    grid : `torch.Tensor`
        positions on the sphere, tensor of shape ``(res_beta, res_alpha, 3)``
    """

    def __init__(self, lmax=None, res=None, normalization='component', dtype=None, device=None):
        super().__init__()

        assert normalization in ['norm', 'component', 'integral'] or torch.is_tensor(normalization), "normalization needs to be 'norm', 'component' or 'integral'"

        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)

        betas, alphas, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=dtype, device=device)

        n = None
        if normalization == 'component':
            # normalize such that all l has the same variance on the sphere
            # given that all componant has mean 0 and variance 1
            n = math.sqrt(4 * math.pi) * betas.new_tensor([
                1 / math.sqrt(2 * l + 1)
                for l in range(lmax + 1)
            ]) / math.sqrt(lmax + 1)
        if normalization == 'norm':
            # normalize such that all l has the same variance on the sphere
            # given that all componant has mean 0 and variance 1/(2L+1)
            n = math.sqrt(4 * math.pi) * betas.new_ones(lmax + 1) / math.sqrt(lmax + 1)
        if normalization == 'integral':
            n = betas.new_ones(lmax + 1)
        if torch.is_tensor(normalization):
            n = normalization
        m = _expand_matrix(range(lmax + 1), dtype=dtype, device=device)  # [l, m, i]
        shb = torch.einsum('lmj,bj,lmi,l->mbi', m, shb, m, n)  # [m, b, i]

        self.lmax, self.res_beta, self.res_alpha = lmax, res_beta, res_alpha
        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    @property
    def grid(self):
        beta, alpha = torch.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        r"""Evaluate

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(..., (l+1)^2)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``[..., beta, alpha]``
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
    """Transform signal on the sphere into spherical tensor

    The inverse transformation of `ToS2Grid`

    Parameters
    ----------
    res : int
    lmax : int
    normalization : {'norm', 'component', 'integral'}
    lmax_in : int, optional
    dtype : torch.dtype or None, optional
    device : torch.device or None, optional

    Examples
    --------

    >>> m = FromS2Grid((100, 101), 6)
    >>> x = torch.randn(3, 100, 101)
    >>> m(x).shape
    torch.Size([3, 49])


    `ToS2Grid` and `FromS2Grid` are inverse of each other

    >>> m = FromS2Grid((100, 101), 6)
    >>> k = ToS2Grid(6, (100, 101))
    >>> x = torch.randn(3, 100, 101)
    >>> x = k(m(x))  # remove high frequencies
    >>> y = k(m(x))
    >>> (x - y).abs().max().item() < 1e-4
    True

    Attributes
    ----------
    grid : `torch.Tensor`
        positions on the sphere, tensor of shape ``(res_beta, res_alpha, 3)``

    """

    def __init__(self, res=None, lmax=None, normalization='component', lmax_in=None, dtype=None, device=None):
        super().__init__()

        assert normalization in ['norm', 'component', 'integral'] or torch.is_tensor(normalization), "normalization needs to be 'norm', 'component' or 'integral'"

        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)

        if lmax_in is None:
            lmax_in = lmax

        betas, alphas, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=dtype, device=device)

        # normalize such that it is the inverse of ToS2Grid
        n = None
        if normalization == 'component':
            n = math.sqrt(4 * math.pi) * betas.new_tensor([
                math.sqrt(2 * l + 1)
                for l in range(lmax + 1)
            ]) * math.sqrt(lmax_in + 1)
        if normalization == 'norm':
            n = math.sqrt(4 * math.pi) * betas.new_ones(lmax + 1) * math.sqrt(lmax_in + 1)
        if normalization == 'integral':
            n = 4 * math.pi * betas.new_ones(lmax + 1)
        if torch.is_tensor(normalization):
            n = normalization
        m = _expand_matrix(range(lmax + 1), dtype=dtype, device=device)  # [l, m, i]
        assert res_beta % 2 == 0
        qw = _quadrature_weights(res_beta // 2, dtype=dtype, device=device) * res_beta**2 / res_alpha  # [b]
        shb = torch.einsum('lmj,bj,lmi,l,b->mbi', m, shb, m, n, qw)  # [m, b, i]

        self.lmax, self.res_beta, self.res_alpha = lmax, res_beta, res_alpha
        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('sha', sha)
        self.register_buffer('shb', shb)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    @property
    def grid(self):
        beta, alpha = torch.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        r"""Evaluate

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``[..., beta, alpha]``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., (l+1)^2)``
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
