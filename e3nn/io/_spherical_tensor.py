from math import pi
from collections import namedtuple

import scipy.signal
import torch

from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid


def _find_peaks_2d(x):
    iii = []
    for i in range(x.shape[0]):
        jj, _ = scipy.signal.find_peaks(x[i, :])
        iii += [(i, j) for j in jj]

    jjj = []
    for j in range(x.shape[1]):
        ii, _ = scipy.signal.find_peaks(x[:, j])
        jjj += [(i, j) for i in ii]

    return list(set(iii).intersection(set(jjj)))


class SphericalTensor(o3.Irreps):
    r"""representation of a signal on the sphere

    A `SphericalTensor` contains the coefficients :math:`A^l` of a function :math:`f` defined on the sphere

    .. math::
        f(x) = \sum_{l=0}^{l_\mathrm{max}} A^l \cdot Y^l(x)


    The way this function is transformed by parity :math:`f \longrightarrow P f` is described by the two parameters :math:`p_v` and :math:`p_a`

    .. math::
        (P f)(x) &= p_v f(p_a x)

        &= \sum_{l=0}^{l_\mathrm{max}} p_v p_a^l A^l \cdot Y^l(x)


    Parameters
    ----------
    lmax : int
        :math:`l_\mathrm{max}`

    p_val : {+1, -1}
        :math:`p_v`

    p_arg : {+1, -1}
        :math:`p_a`


    Examples
    --------

    >>> SphericalTensor(3, 1, 1)
    1x0e+1x1e+1x2e+1x3e

    >>> SphericalTensor(3, 1, -1)
    1x0e+1x1o+1x2e+1x3o
    """
    def __new__(cls, lmax, p_val, p_arg):
        return super().__new__(cls, [(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])

    def with_peaks_at(self, vectors, values=None):
        r"""Create a spherical tensor with peaks

        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(N, 3)``

        values : `torch.Tensor`, optional
            value on the peak, tensor of shape ``(N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(self.dim,)``

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [3.0, 4.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> s.signal_xyz(x, pos)
        tensor([1.0000, 5.0000])

        >>> val = torch.tensor([
        ...     -1.5,
        ...     2.0,
        ... ])
        >>> x = s.with_peaks_at(pos, val)
        >>> s.signal_xyz(x, pos)
        tensor([-1.5000,  2.0000])
        """
        if values is not None:
            vectors, values = torch.broadcast_tensors(vectors, values[..., None])
            values = values[..., 0]

        # empty set of vectors returns a 0 spherical tensor
        if vectors.numel() == 0:
            return torch.zeros(vectors.shape[:-2] + (self.dim,))

        assert self[0][1].p == 1, "since the value is set by the radii who is even, p_val has to be 1"

        assert vectors.dim() == 2 and vectors.shape[1] == 3

        if values is None:
            values = vectors.norm(dim=1)  # [batch]
        vectors = vectors[values != 0]  # [batch, 3]
        values = values[values != 0]

        coeff = o3.spherical_harmonics(self, vectors, normalize=True)  # [batch, l * m]
        A = torch.einsum(
            "ai,bi->ab",
            coeff,
            coeff
        )
        # Y(v_a) . Y(v_b) solution_b = radii_a
        solution = torch.lstsq(values, A).solution.reshape(-1)  # [b]
        assert (values - A @ solution).abs().max() < 1e-5 * values.abs().max()

        return solution @ coeff

    def sum_of_diracs(self, positions, values):
        r"""Sum (almost-) dirac deltas

        .. math::

            f(x) = \sum_i v_i \delta^L(\vec r_i)

        where :math:`\delta^L` is the apporximation of a dirac delta.

        Parameters
        ----------
        positions : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            :math:`v_i` tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(7, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ... ])
        >>> val = torch.tensor([
        ...     -1.0,
        ...     1.0,
        ... ])
        >>> x = s.sum_of_diracs(pos, val)
        >>> s.signal_xyz(x, torch.eye(3)).mul(10.0).round()
        tensor([-10.,  10.,  -0.])

        >>> s.sum_of_diracs(torch.empty(1, 0, 2, 3), torch.empty(2, 0, 1)).shape
        torch.Size([2, 0, 64])

        >>> s.sum_of_diracs(torch.randn(1, 3, 2, 3), torch.randn(2, 1, 1)).shape
        torch.Size([2, 3, 64])
        """
        positions, values = torch.broadcast_tensors(positions, values[..., None])
        values = values[..., 0]

        if positions.numel() == 0:
            return torch.zeros(values.shape[:-1] + (self.dim,))

        y = o3.spherical_harmonics(self, positions, True)  # [..., N, dim]
        v = values[..., None]

        return 4 * pi / (self.lmax + 1)**2 * (y * v).sum(-2)

    def from_samples_on_s2(self, positions, values, res=100):
        r"""Convert a set of position on the sphere and values into a spherical tensor

        Parameters
        ----------
        positions : `torch.Tensor`
            tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(2, 1, 1)
        >>> pos = torch.tensor([
        ...     [
        ...         [0.0, 0.0, 1.0],
        ...         [0.0, 0.0, -1.0],
        ...     ],
        ...     [
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, -1.0, 0.0],
        ...     ],
        ... ], dtype=torch.float64)
        >>> val = torch.tensor([
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ... ], dtype=torch.float64)
        >>> s.from_samples_on_s2(pos, val, res=200).long()
        tensor([[0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0]])

        >>> pos = torch.empty(2, 0, 10, 3)
        >>> val = torch.empty(2, 0, 10)
        >>> s.from_samples_on_s2(pos, val)
        tensor([], size=(2, 0, 9))

        """
        positions, values = torch.broadcast_tensors(positions, values[..., None])
        values = values[..., 0]

        if positions.numel() == 0:
            return torch.zeros(values.shape[:-1] + (self.dim,))

        positions = torch.nn.functional.normalize(positions, dim=-1)  # forward 0's instead of nan for zero-radius

        size = positions.shape[:-2]
        n = positions.shape[-2]
        positions = positions.reshape(-1, n, 3)
        values = values.reshape(-1, n)

        s2 = FromS2Grid(res=res, lmax=self.lmax, normalization='integral', dtype=values.dtype, device=values.device)
        pos = s2.grid.reshape(1, -1, 3)

        cd = torch.cdist(pos, positions)  # [batch, b*a, N]
        i = torch.arange(len(values)).view(-1, 1)  # [batch, 1]
        j = cd.argmin(2)  # [batch, b*a]
        val = values[i, j]  # [batch, b*a]
        val = val.reshape(*size, s2.res_beta, s2.res_alpha)

        return s2(val)

    def norms(self, signal):
        r"""The norms of each l component

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(..., dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., lmax+1)``

        Examples
        --------
        Examples
        --------
        >>> s = SphericalTensor(1, 1, -1)
        >>> s.norms(torch.tensor([1.5, 0.0, 3.0, 4.0]))
        tensor([1.5000, 5.0000])
        """
        i = 0
        norms = []
        for _, ir in self:
            norms += [signal[..., i: i + ir.dim].norm(dim=-1)]
            i += ir.dim
        return torch.stack(norms, dim=-1)

    def signal_xyz(self, signal, r):
        r"""Evaluate the signal on given points on the sphere

        .. math::

            f(\vec x / \|\vec x\|)

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(*A, self.dim)``

        r : `torch.Tensor`
            tensor of shape ``(*B, 3)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(*A, *B)``

        Examples
        --------
        >>> s = SphericalTensor(3, 1, -1)
        >>> s.signal_xyz(s.randn(2, 1, 3, -1), torch.randn(2, 4, 3)).shape
        torch.Size([2, 1, 3, 2, 4])
        """
        sh = o3.spherical_harmonics(self, r, normalize=True)
        dim = (self.lmax + 1)**2
        output = torch.einsum('bi,ai->ab', sh.reshape(-1, dim), signal.reshape(-1, dim))
        return output.reshape(signal.shape[:-1] + r.shape[:-1])

    def signal_on_grid(self, signal, res=100, normalization='integral'):
        r"""Evaluate the signal on a grid on the sphere
        """
        Ret = namedtuple("Return", "grid, values")
        s2 = ToS2Grid(lmax=self.lmax, res=res, normalization=normalization)
        return Ret(s2.grid, s2(signal))

    def plotly_surface(self, signals, centers=None, res=100, radius=True, relu=False, normalization='integral'):
        r"""Create traces for plotly

        Examples
        --------
        >>> import plotly.graph_objects as go
        >>> x = SphericalTensor(4, +1, +1)
        >>> traces = x.plotly_surface(x.randn(-1))
        >>> traces = [go.Surface(**d) for d in traces]
        >>> fig = go.Figure(data=traces)
        """
        signals = signals.reshape(-1, self.dim)

        if centers is None:
            centers = [None] * len(signals)
        else:
            centers = centers.reshape(-1, 3)

        traces = []
        for signal, center in zip(signals, centers):
            r, f = self.plot(signal, center, res, radius, relu, normalization)
            traces += [dict(
                x=r[:, :, 0].numpy(),
                y=r[:, :, 1].numpy(),
                z=r[:, :, 2].numpy(),
                surfacecolor=f.numpy(),
            )]
        return traces

    def plot(self, signal, center=None, res=100, radius=True, relu=False, normalization='integral'):
        r"""Create surface in order to make a plot
        """
        assert signal.dim() == 1

        r, f = self.signal_on_grid(signal, res, normalization)
        f = f.relu() if relu else f

        # beta: [0, pi]
        r[0] = r.new_tensor([0.0, 1.0, 0.0])
        r[-1] = r.new_tensor([0.0, -1.0, 0.0])
        f[0] = f[0].mean()
        f[-1] = f[-1].mean()

        # alpha: [0, 2pi]
        r = torch.cat([r, r[:, :1]], dim=1)  # [beta, alpha, 3]
        f = torch.cat([f, f[:, :1]], dim=1)  # [beta, alpha]

        if radius:
            r *= f.abs().unsqueeze(-1)

        if center is not None:
            r += center

        return r, f

    def find_peaks(self, signal, res=100):
        r"""Locate peaks on the sphere

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [4.0, 0.0, 4.0],
        ...     [0.0, 5.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> pos, val = s.find_peaks(x)
        >>> pos[val > 4.0].mul(10).round().abs()
        tensor([[ 7.,  0.,  7.],
                [ 0., 10.,  0.]])
        >>> val[val > 4.0].mul(10).round().abs()
        tensor([57., 50.])
        """
        x1, f1 = self.signal_on_grid(signal, res)

        abc = torch.tensor([pi / 2, pi / 2, pi / 2])
        R = o3.angles_to_matrix(*abc)
        D = self.D_from_matrix(R)

        r_signal = D @ signal
        rx2, f2 = self.signal_on_grid(r_signal, res)
        x2 = torch.einsum('ij,baj->bai', R.T, rx2)

        ij = _find_peaks_2d(f1)
        x1p = torch.stack([x1[i, j] for i, j in ij])
        f1p = torch.stack([f1[i, j] for i, j in ij])

        ij = _find_peaks_2d(f2)
        x2p = torch.stack([x2[i, j] for i, j in ij])
        f2p = torch.stack([f2[i, j] for i, j in ij])

        # Union of the results
        mask = torch.cdist(x1p, x2p) < 2 * pi / res
        x = torch.cat([x1p[mask.sum(1) == 0], x2p])
        f = torch.cat([f1p[mask.sum(1) == 0], f2p])

        return x, f
