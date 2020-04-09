# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import math

import scipy.signal
import torch

from e3nn import o3, rsh


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
