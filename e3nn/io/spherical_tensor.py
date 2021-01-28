from math import pi

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

    .. math::
        f(x) = \sum_{l=0}^{l_\mathrm{max}} A^l \cdot Y^l(x)

        (P f)(x) = p_v f(p_a x)

        (P f)(x) = \sum_{l=0}^{l_\mathrm{max}} p_v p_a^l A^l \cdot Y^l(x)

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

    def from_geometry_adjusted(self, vectors):
        r"""Convert a set of relative positions into a spherical tensor

        TODO rename this function?

        Examples
        --------
        >>> x = SphericalTensor(4, 1, -1)
        >>> p = torch.tensor([
        ...     [1.0, 0, 0],
        ...     [3.0, 4.0, 0],
        ... ])
        >>> d = x.from_geometry_adjusted(p)
        >>> x.signal_xyz(d, p)
        tensor([1.0000, 5.0000])
        """
        assert self[0][1].p == 1, "since the value is set by the radii who is even, p_val has to be 1"
        vectors = vectors.reshape(-1, 3)
        radii = vectors.norm(dim=1)  # [batch]
        vectors = vectors[radii > 0]  # [batch, 3]

        coeff = o3.spherical_harmonics(self, vectors, normalize=True)  # [batch, l * m]
        A = torch.einsum(
            "ai,bi->ab",
            coeff,
            coeff
        )
        # Y(v_a) . Y(v_b) solution_b = radii_a
        solution = torch.lstsq(radii, A).solution.reshape(-1)  # [b]
        assert (radii - A @ solution).abs().max() < 1e-5 * radii.abs().max()

        return solution @ coeff

    def from_geometry_global_rescale(self, vectors):
        r"""Convert a set of relative positions into a spherical tensor

        TODO rename this function?
        """
        assert self[0][1].p == 1, "since the value is set by the radii who is even, p_val has to be 1"
        vectors = vectors.reshape(-1, 3)
        radii = vectors.norm(dim=1)
        sh = o3.spherical_harmonics(self, vectors, normalize=True)
        # 0.5 * sum_a ( Y(v_a) . sum_b r_b Y(v_b) s - r_a )^2
        A = torch.einsum('ai,b,bi->a', sh, radii, sh)
        # 0.5 * sum_a ( A_a s - r_a )^2
        # sum_a A_a^2 s = sum_a A_a r_a
        s = torch.dot(A, radii) / A.norm().pow(2)
        return s * torch.einsum('a,ai->i', radii, sh)

    def from_samples_on_s2(self, positions, values, res=100):
        r"""Convert a set of position on the sphere and values into a spherical tensor

        TODO rename this function?
        """
        positions = positions.reshape(-1, 3)
        values = values.reshape(-1)
        positions /= positions.norm(dim=1, keepdim=True)
        assert positions.shape[0] == values.shape[0], "positions and values must have the same number of points"

        s2 = FromS2Grid(res=res, lmax=self.lmax, normalization='integral')
        pos = s2.grid

        cd = torch.cdist(pos, positions)
        val = values[cd.argmin(2)]

        return s2(val)

    def norms(self, signal):
        r"""The norms of each l component
        """
        i = 0
        norms = []
        for _, ir in self:
            norms += [signal[..., i: i + ir.dim].norm(dim=-1)]
            i += ir.dim
        return torch.stack(norms, dim=-1)

    def signal_xyz(self, signal, r):
        r"""Evaluate the signal on given points on the sphere
        """
        sh = o3.spherical_harmonics(self, r, normalize=True)
        dim = (self.lmax + 1)**2
        output = torch.einsum('ai,zi->za', sh.reshape(-1, dim), signal.reshape(-1, dim))
        return output.reshape(signal.shape[:-1] + r.shape[:-1])

    def signal_on_grid(self, signal, res=100, normalization='integral'):
        r"""Evaluate the signal on a grid on the sphere
        """
        s2 = ToS2Grid(lmax=self.lmax, res=res, normalization=normalization)
        return s2.grid, s2(signal)

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
        r[0] = r.new_tensor([0, 0, 1])
        r[-1] = r.new_tensor([0, 0, -1])
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
        """
        x1, f1 = self.signal_on_grid(signal, res)

        abc = pi / 2, pi / 2, pi / 2
        R = o3.rot(*abc)
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
