# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import math
import numpy as np

import scipy.signal
import torch

from e3nn import o3, rsh, rs
from e3nn.s2grid import ToS2Grid, s2_grid
from e3nn.kernel_mod import FrozenKernel


class SphericalHarmonicsProject(torch.nn.Module):
    def __init__(self, alpha, beta, lmax):
        super().__init__()
        sh = torch.cat([rsh.spherical_harmonics(l, alpha, beta) for l in range(lmax + 1)])
        self.register_buffer("sh", sh)

    def forward(self, coeff):
        return torch.einsum("i,i...->...", (coeff, self.sh))


class SphericalHarmonicsFindPeaks(torch.nn.Module):
    def __init__(self, n, lmax):
        super().__init__()
        self.n = n
        self.lmax = lmax

        R = o3.rot(math.pi / 2, math.pi / 2, math.pi / 2)
        self.xyz1, self.proj1 = self.precompute(R)

        R = o3.rot(0, 0, 0)
        self.xyz2, self.proj2 = self.precompute(R)

    def precompute(self, R):
        a = torch.linspace(0, 2 * math.pi, 2 * self.n)
        b = torch.linspace(0, math.pi, self.n)[2:-2]
        a, b = torch.meshgrid(a, b)

        xyz = torch.stack(o3.angles_to_xyz(a, b), dim=-1) @ R.t()
        a, b = o3.xyz_to_angles(xyz)

        proj = SphericalHarmonicsProject(a, b, self.lmax)
        return xyz, proj

    def detect_peaks(self, signal, xyz, proj):
        f = proj(signal)

        beta_pass = []
        for i in range(f.size(0)):
            jj, _ = scipy.signal.find_peaks(f[i])
            beta_pass += [(i, j) for j in jj]

        alpha_pass = []
        for j in range(f.size(1)):
            ii, _ = scipy.signal.find_peaks(f[:, j])
            alpha_pass += [(i, j) for i in ii]

        peaks = list(set(beta_pass).intersection(set(alpha_pass)))

        radius = torch.stack([f[i, j] for i, j in peaks]) if peaks else torch.empty(0)
        peaks = torch.stack([xyz[i, j] for i, j in peaks]) if peaks else torch.empty(0, 3)
        return peaks, radius

    def forward(self, signal):
        peaks1, radius1 = self.detect_peaks(signal, self.xyz1, self.proj1)
        peaks2, radius2 = self.detect_peaks(signal, self.xyz2, self.proj2)

        diff = peaks1.unsqueeze(1) - peaks2.unsqueeze(0)
        mask = diff.norm(dim=-1) < 2 * math.pi / self.n

        peaks = torch.cat([peaks1[mask.sum(1) == 0], peaks2])
        radius = torch.cat([radius1[mask.sum(1) == 0], radius2])

        return peaks, radius


def spherical_harmonics_dirac(lmax, alpha, beta):
    """
    approximation of a signal that is 0 everywhere except on the angle (alpha, beta) where it is one.
    the higher is lmax the better is the approximation
    """
    ls = list(range(lmax + 1))
    a = sum(2 * l + 1 for l in ls) / (4 * math.pi)
    return rsh.spherical_harmonics_alpha_beta(ls, alpha, beta) / a


def spherical_harmonics_coeff_to_sphere(coeff, alpha, beta):
    """
    Evaluate the signal on the sphere
    """
    lmax = round(coeff.shape[-1] ** 0.5) - 1
    ls = list(range(lmax + 1))
    sh = rsh.spherical_harmonics_alpha_beta(ls, alpha, beta)
    return torch.einsum('...i,i->...', sh, coeff)


def projection(vectors, lmax, sum_points=True, radius=True):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.]

    if radius:
        radii = radii[radii > 0.]
    else:
        radii = torch.ones_like(radii[radii > 0.])

    angles = o3.xyz_to_angles(vectors)
    coeff = spherical_harmonics_dirac(lmax, *angles)
    coeff *= radii.unsqueeze(-1)
    return coeff.sum(-2) if sum_points else coeff


def adjusted_projection(vectors, lmax, sum_points=True, radius=True):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.]
    angles = o3.xyz_to_angles(vectors)

    coeff = projection(vectors, lmax, sum_points=False, radius=radius)
    A = torch.einsum("ai,bi->ab", (rsh.spherical_harmonics_alpha_beta(list(range(lmax + 1)), *angles), coeff))
    try:
        coeff *= torch.lstsq(radii, A).solution.view(-1).unsqueeze(-1)
    except:
        coeff *= torch.gels(radii, A).solution.view(-1).unsqueeze(-1)
    return coeff.sum(-2) if sum_points else coeff


class SphericalTensor():
    def __init__(self, signal, mul, lmax):
        self.signal = signal
        self.lmax = lmax
        self.mul = mul
        self.Rs = [(mul, l) for l in range(lmax + 1)]

    @classmethod
    def from_geometry(cls, vectors, lmax, sum_points=True, radius=True):
        signal = adjusted_projection(vectors, lmax, sum_points=sum_points, radius=radius)
        return cls(signal, 1, lmax)

    @classmethod
    def from_geometry_with_radial(cls, vectors, radial_model, lmax, sum_points=True):
        r = vectors.norm(2, -1)
        radial_functions = radial_model(r)
        _N, R = radial_functions.shape
        Rs = [(R, L) for L in range(lmax + 1)]
        mul_map = rs.map_mul_to_Rs(Rs)
        radial_functions = torch.einsum('nr,dr->nd',
                                        radial_functions.repeat(1, lmax + 1),
                                        mul_map)  # [N, signal]

        Ys = projection(vectors, lmax, sum_points=False, radius=False)  # [channels, N]
        irrep_map = rs.map_irrep_to_Rs(Rs)
        Ys = torch.einsum('nc,dc->nd', Ys, irrep_map)  # [N, signal]

        signal = Ys * radial_functions  # [N, signal]

        if sum_points:
            signal = signal.sum(0)
        new_cls = cls(signal, R, lmax)
        new_cls.radial_model = radial_model
        return new_cls

    def sph_norm(self):
        Rs = self.Rs
        signal = self.signal
        n_mul = sum([mul for mul, L in Rs])
        # Keep shape after Rs the same
        norms = torch.zeros(n_mul, *signal.shape[1:])
        sig_index = 0
        norm_index = 0
        for mul, L in Rs:
            for _ in range(mul):
                norms[norm_index] = signal[sig_index: sig_index + (2 * L + 1)].norm(2, 0)
                norm_index += 1
                sig_index += 2 * L + 1
        return norms

    def signal_on_sphere(self, n=100):
        n_mul = sum([mul for mul, L in self.Rs])
        # May want to consider caching this object in SphericalTensor
        grid = ToS2Grid(self.lmax, res=n)
        res_beta, res_alpha = grid.res_alpha, grid.res_beta
        betas, alphas = s2_grid(res_beta, res_alpha)
        beta, alpha = torch.meshgrid(betas, alphas)
        x, y, z = o3.angles_to_xyz(alpha, beta)
        r = torch.stack([x, y, z], dim=-1)
        return r, grid(self.signal)

    def plot(self, n=100, radius=True, center=None, relu=True):
        """
        surface = self.plot()
        fig = go.Figure(data=[surface])
        fig.show()
        """
        r, f = self.signal_on_sphere(n)
        f = f.relu() if relu else f

        r = torch.cat([r, r[:, 0].unsqueeze(1)], dim=1)
        f = torch.cat([f, f[:, 0].unsqueeze(1)], dim=1)

        if radius:
            r *= f.abs().unsqueeze(-1)

        if center:
            r += center.unsqueeze(0)

        return r, f

    def plot_with_radial(self, box_length, center=None,
                         sh=rsh.spherical_harmonics_xyz, n=30,
                         radial_model=None, relu=True):
        muls, _Ls = zip(*self.Rs)
        # We assume radial functions are repeated across L's
        assert len(set(muls)) == 1
        num_L = len(self.Rs)
        if radial_model is None:
            radial_model = self.radial_model

        def new_radial(x):
            return radial_model(x).repeat(1, num_L)  # Repeat along filter dim
        r, f = plot_on_grid(box_length, new_radial, self.Rs, sh=sh, n=n)
        # Multiply coefficients
        f = torch.einsum('xd,d->x', f, self.signal)
        f = f.relu() if relu else f

        if center:
            r += center.unsqueeze(0)

        return r, f

    def find_peaks(self, which_mul=None, n=100, min_radius=0.1,
                   percentage=False, absolute_min=0.1, radius=True):
        if not hasattr(self, 'peak_finder') or self.peak_finder.n != n:
            lmax = max(L for mult, L in self.Rs)
            self.peak_finder = SphericalHarmonicsFindPeaks(n, lmax)

        peaks, radius = self.peak_finder.forward(self.signal)

        if percentage:
            self.used_radius = max((min_radius * torch.max(radius)),
                                   absolute_min)
            keep_indices = (radius > max((min_radius * torch.max(radius)),
                                         absolute_min))
        else:
            self.used_radius = min_radius
            keep_indices = (radius > min_radius)
        return peaks[keep_indices] * radius[keep_indices].unsqueeze(-1)

    def __add__(self, other):
        if self.Rs == other.Rs:
            from copy import deepcopy
            return SphericalTensor(self.signal + other.signal,
                                   deepcopy(self.Rs))

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        # Add check for feature_Rs.
        if self.Rs == other.Rs:
            dot = (self.signal * other.signal).sum(-1)
            dot /= (self.signal.norm(2, 0) * other.signal.norm(2, 0))
            return dot

    def __matmul__(self, other):
        # Tensor product
        # Assume first index is Rs
        # Better handle mismatch of features indices
        Rs_out, C = rs.tensor_product(self.Rs, other.Rs)
        Rs_out = [(mult, L) for mult, L, parity in Rs_out]
        new_signal = torch.einsum('kij,i...,j...->k...',
                                  (C, self.signal, other.signal))
        return SphericalTensor(new_signal, Rs_out)

    def __rmatmul__(self, other):
        # Tensor product
        return self.__matmul__(other)


def plot_on_grid(box_length, radial_model, Rs, sh=rsh.spherical_harmonics_xyz, n=30):
    L_to_index = {}
    set_of_L = set([L for mul, L in Rs])
    start = 0
    for L in set_of_L:
        L_to_index[L] = [start, start + 2 * L + 1]
        start += 2 * L + 1

    r = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j].reshape(3, -1)
    r = r.transpose(1, 0)
    r *= box_length / 2.
    r = torch.from_numpy(r)

    Rs_in = [(1, 0)]
    Rs_out = Rs
    radial_lambda = lambda x: radial_model
    grid = FrozenKernel(Rs_in, Rs_out, radial_lambda, r, sh=sh)
    R = grid.R(grid.radii)
    # j is just 1 because Rs_in is 1d
    f = torch.einsum('xjmw,xw->xj', grid.Q, R)
    return r, f