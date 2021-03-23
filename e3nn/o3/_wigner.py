r"""Core functions of :math:`SO(3)`
"""
import os

import torch

from e3nn import o3
from e3nn.util import explicit_default_types

_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))
# _Jd is a list of tensors of shape (2l+1, 2l+1)
# _W3j_flat is a flatten version of W3j symbols
# _W3j_indices is a dict from (l1, l2, l3) -> slice(i, j) to index the flat tensor
# only l1 <= l2 <= l3 are stored


def _z_rot_mat(angle, l):
    r"""
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Code of this function has beed copied from `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn>`_ made by Taco Cohen.

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    if not l < len(_Jd):
        raise NotImplementedError(f'wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more')

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def wigner_3j(l1, l2, l3, flat_src=_W3j_flat, dtype=None, device=None):
    r"""Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:

        .. math::

            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

        where :math:`D` are given by `wigner_D`.

        .. math::

            C_{ijk} C_{ijk} = 1

    Parameters
    ----------
    l1 : int
        :math:`l_1`

    l2 : int
        :math:`l_2`

    l3 : int
        :math:`l_3`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensor. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensor. If ``None`` then set to the default device of the current context.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3

    try:
        if l1 <= l2 <= l3:
            out = flat_src[_W3j_indices[(l1, l2, l3)]].reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1).clone()
        if l1 <= l3 <= l2:
            out = flat_src[_W3j_indices[(l1, l3, l2)]].reshape(2 * l1 + 1, 2 * l3 + 1, 2 * l2 + 1).transpose(1, 2).mul((-1) ** (l1 + l2 + l3)).clone()
        if l2 <= l1 <= l3:
            out = flat_src[_W3j_indices[(l2, l1, l3)]].reshape(2 * l2 + 1, 2 * l1 + 1, 2 * l3 + 1).transpose(0, 1).mul((-1) ** (l1 + l2 + l3)).clone()
        if l3 <= l2 <= l1:
            out = flat_src[_W3j_indices[(l3, l2, l1)]].reshape(2 * l3 + 1, 2 * l2 + 1, 2 * l1 + 1).transpose(0, 2).mul((-1) ** (l1 + l2 + l3)).clone()
        if l2 <= l3 <= l1:
            out = flat_src[_W3j_indices[(l2, l3, l1)]].reshape(2 * l2 + 1, 2 * l3 + 1, 2 * l1 + 1).transpose(0, 2).transpose(1, 2).clone()
        if l3 <= l1 <= l2:
            out = flat_src[_W3j_indices[(l3, l1, l2)]].reshape(2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1).transpose(0, 2).transpose(0, 1).clone()
    except KeyError:
        raise NotImplementedError(f'Wigner 3j symbols maximum l implemented is {max(_W3j_indices.keys())[0]}, send us an email to ask for more')

    dtype, device = explicit_default_types(dtype, device)
    return out.to(dtype=dtype, device=device)


def _generate_wigner_3j(l1, l2, l3, dtype=None, device=None):  # pragma: no cover
    r"""Computes the 3-j symbol
    """
    # these three propositions are equivalent
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert abs(l3 - l1) <= l2 <= l3 + l1
    assert abs(l1 - l2) <= l3 <= l1 + l2

    n = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)  # dimension of the 3-j symbol

    def _DxDxD(a, b, c):
        D1 = wigner_D(l1, a, b, c)
        D2 = wigner_D(l2, a, b, c)
        D3 = wigner_D(l3, a, b, c)
        return torch.einsum('il,jm,kn->ijklmn', D1, D2, D3).reshape(n, n)

    random_angles = torch.tensor([
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.29089583, 3.90040975],
    ], dtype=dtype, device=device)

    B = random_angles.new_zeros((n, n))
    for abc in random_angles:
        D = _DxDxD(*abc) - torch.eye(n, dtype=dtype, device=device)
        B += D.T @ D

    eigenvalues, eigenvectors = torch.symeig(B, eigenvectors=True)
    assert eigenvalues[0] < 1e-10
    Q = eigenvectors[:, 0]
    assert (B @ Q).norm() < 1e-10
    Q = Q.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)

    Q[Q.abs() < 1e-14] = 0

    if Q[l1, l2, l3] != 0:
        if Q[l1, l2, l3] < 0:
            Q.neg_()
    else:
        if next(x for x in Q.flatten() if x != 0) < 0:
            Q.neg_()

    abc = o3.rand_angles(100, dtype=dtype, device=device)
    Q2 = torch.einsum("zil,zjm,zkn,lmn->zijk", wigner_D(l1, *abc), wigner_D(l2, *abc), wigner_D(l3, *abc), Q)
    assert (Q - Q2).norm() < 1e-10
    assert abs(Q.norm() - 1) < 1e-10

    return Q
