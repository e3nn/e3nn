# pylint: disable=no-member, bare-except, invalid-name, missing-docstring, line-too-long
import numpy as np
import torch
import math

import e3nn
import e3nn.o3 as o3
import e3nn.rs as rs
from e3nn.util.cache_file import cached_dirpklgz
import e3nn.util.plot as plot
from e3nn.spherical_harmonics import SphericalHarmonicsFindPeaks

__authors__ = "Tess E. Smidt, Mario Geiger, Josh Rackers"

torch.set_default_dtype(torch.float64)


def direct_sum(*matrices):
    # Slight modification of se3cnn.SO3.direct_sum
    """
    Direct sum of matrices, put them in the diagonal
    """

    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(*total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out


def projection(vectors, L_max, sum_points=True, radius=True):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.]

    if radius:
        radii = radii[radii > 0.]
    else:
        radii = torch.ones_like(radii[radii > 0.])

    angles = e3nn.o3.xyz_to_angles(vectors)
    coeff = e3nn.o3.spherical_harmonics_dirac(L_max, *angles)
    coeff *= radii.unsqueeze(-2)
    return coeff.sum(-1) if sum_points else coeff


def adjusted_projection(vectors, L_max, sum_points=True, radius=True):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.]
    angles = e3nn.o3.xyz_to_angles(vectors)

    coeff = projection(vectors, L_max, sum_points=False, radius=radius)

    A = torch.einsum("ia,ib->ab", (e3nn.o3.spherical_harmonics(list(range(L_max + 1)), *angles), coeff))
    try:
        coeff *= torch.lstsq(radii, A).solution.view(-1)
    except:
        coeff *= torch.gels(radii, A).solution.view(-1)
    return coeff.sum(-1) if sum_points else coeff


class SphericalTensor():
    def __init__(self, tensor, Rs):
        self.tensor = tensor
        self.Rs = Rs

    @classmethod
    def from_geometry(cls, vectors, L_max, sum_points=True, radius=True):
        Rs = [(1, L) for L in range(L_max + 1)]
        tensor = adjusted_projection(vectors, L_max, sum_points=sum_points, radius=radius)
        return cls(tensor, Rs)

    @classmethod
    def from_geometry_with_radial(cls, vectors, radial_model, L_max, sum_points=True):
        r = vectors.norm(2, -1)
        radial_functions = radial_model(r)
        _N, R = radial_functions.shape
        Rs = [(R, L) for L in range(L_max + 1)]
        mul_map = rs.map_mul_to_Rs(Rs)
        radial_functions = torch.einsum('nr,dr->nd',
                                        radial_functions.repeat(1, L_max + 1),
                                        mul_map)  # [N, tensor]

        Ys = projection(vectors, L_max, sum_points=False, radius=False)  # [channels, N]
        irrep_map = rs.map_irrep_to_Rs(Rs)
        Ys = torch.einsum('cn,dc->nd', Ys, irrep_map)  # [N, tensor]

        tensor = Ys * radial_functions  # [N, tensor]

        if sum_points:
            tensor = tensor.sum(0)
        new_cls = cls(tensor, Rs)
        new_cls.radial_model = radial_model
        return new_cls

    def sph_norm(self):
        Rs = self.Rs
        tensor = self.tensor
        n_mul = sum([mul for mul, L in Rs])
        # Keep shape after Rs the same
        norms = torch.zeros(n_mul, *tensor.shape[1:])
        sig_index = 0
        norm_index = 0
        for mul, L in Rs:
            for _ in range(mul):
                norms[norm_index] = tensor[sig_index: sig_index + (2 * L + 1)].norm(2, 0)
                norm_index += 1
                sig_index += 2 * L + 1
        return norms

    def tensor_on_sphere(self, which_mul=None, n=100, radius=True):
        n_mul = sum([mul for mul, L in self.Rs])
        if which_mul:
            if len(which_mul) != n_mul:
                raise ValueError("which_mul and number of multiplicities is not equal.")
        else:
            which_mul = [1 for i in range(n_mul)]

        # Need to handle if tensor is featurized
        x, y, z = (None, None, None)
        Ys = []
        for mul, L in self.Rs:
            # Using cache-able function
            x, y, z, Y = spherical_harmonics_on_grid(L, n)
            Ys += [Y] * mul

        f = self.tensor.unsqueeze(1).unsqueeze(2) * torch.cat(Ys, dim=0)
        f = f.sum(0)
        return x, y, z, f

    def plot(self, which_mul=None, n=100, radius=True, center=None, relu=True):
        """
        surface = self.plot()
        fig = go.Figure(data=[surface])
        fig.show()
        """
        import plotly.graph_objs as go

        x, y, z, f = self.tensor_on_sphere(which_mul, n, radius)
        f = f.relu() if relu else f

        if radius:
            r = f.abs()
            x = x * r
            y = y * r
            z = z * r

        if center is not None:
            x = x + center[0]
            y = y + center[1]
            z = z + center[2]

        return go.Surface(x=x.numpy(), y=y.numpy(), z=z.numpy(), surfacecolor=f.numpy())

    def plot_with_radial(self, box_length, center=None,
                         sh=o3.spherical_harmonics_xyz, n=30,
                         radial_model=None, relu=True):
        muls, _Ls = zip(*self.Rs)
        # We assume radial functions are repeated across L's
        assert len(set(muls)) == 1
        num_L = len(self.Rs)
        if radial_model is None:
            radial_model = self.radial_model

        def new_radial(x):
            return radial_model(x).repeat(1, num_L)  # Repeat along filter dim
        r, f = plot_data_on_grid(box_length, new_radial, self.Rs, sh=sh, n=n)
        # Multiply coefficients
        f = torch.einsum('xd,d->x', f, self.tensor)
        f = f.relu() if relu else f

        if center is not None:
            r += center.unsqueeze(0)

        return r, f

    def wigner_D_on_grid(self, n):
        try:
            return getattr(self, "wigner_D_grid_{}".format(n))
        except:
            blocks = [wigner_D_on_grid(L, n)
                      for mul, L in self.Rs for m in range(mul)]
            wigner_D = direct_sum(*blocks)
            setattr(self, "wigner_D_grid_{}".format(n), wigner_D)
            return getattr(self, "wigner_D_grid_{}".format(n))

    def cross_correlation(self, other, n, normalize=True):
        if self.Rs != other.Rs:
            raise ValueError("Rs must match")
        wigner_D = self.wigner_D_on_grid(n)
        normalize_by = (self.tensor.norm(2, 0) * other.tensor.norm(2, 0))
        cross_corr = torch.einsum(
            'abcji,j,i->abc', (wigner_D, self.tensor, other.tensor)
        )
        return cross_corr / normalize_by if normalize else cross_corr

    def find_peaks(self, which_mul=None, n=100, min_radius=0.1,
                   percentage=False, absolute_min=0.1, radius=True):
        if not hasattr(self, 'peak_finder') or self.peak_finder.n != n:
            L_max = max(L for mult, L in self.Rs)
            self.peak_finder = SphericalHarmonicsFindPeaks(n, L_max)

        peaks, radius = self.peak_finder.forward(self.tensor)

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
            return SphericalTensor(self.tensor + other.tensor,
                                   deepcopy(self.Rs))

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        # Add check for feature_Rs.
        if self.Rs == other.Rs:
            dot = (self.tensor * other.tensor).sum(-1)
            dot /= (self.tensor.norm(2, 0) * other.tensor.norm(2, 0))
            return dot

    def __matmul__(self, other):
        # Tensor product
        # Assume first index is Rs
        # Better handle mismatch of features indices
        Rs_out, C = e3nn.rs.tensor_product(self.Rs, other.Rs)
        Rs_out = [(mult, L) for mult, L, parity in Rs_out]
        new_tensor = torch.einsum('kij,i...,j...->k...',
                                  (C, self.tensor, other.tensor))
        return SphericalTensor(new_tensor, Rs_out)

    def __rmatmul__(self, other):
        # Tensor product
        return self.__matmul__(other)


def plot_data_on_grid(box_length, radial, Rs, sh=o3.spherical_harmonics_xyz,
                      n=30):
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
    Ys = sh(set_of_L, r)
    R = radial(r.norm(2, -1)).detach()  # [r_values, n_r_filters]
    assert R.shape[-1] == rs.mul_dim(Rs)

    R_helper = torch.zeros(R.shape[-1], rs.dim(Rs))
    Ys_indices = []
    for mul, L in Rs:
        Ys_indices += list(range(L_to_index[L][0], L_to_index[L][1])) * mul

    R_helper = rs.map_mul_to_Rs(Rs)

    full_Ys = Ys[Ys_indices]  # [values, rs.dim(Rs)]]
    full_Ys = full_Ys.reshape(full_Ys.shape[0], -1)

    all_f = torch.einsum('xn,dn,dx->xd', R, R_helper, full_Ys)
    return r, all_f


@cached_dirpklgz("cache/euler_grids")
def euler_angles_on_grid(n):
    alpha = torch.linspace(0, 2 * math.pi, 2 * n)
    beta = torch.linspace(0, math.pi, n)
    gamma = torch.linspace(0, 2 * math.pi, 2 * n)
    alpha, beta, gamma = torch.meshgrid(alpha, beta, gamma)
    return alpha, beta, gamma


def spherical_surface(n):
    alpha = torch.linspace(0, 2 * math.pi, 2 * n)
    beta = torch.linspace(0, math.pi, 2 * n)
    beta, alpha = torch.meshgrid(beta, alpha)
    x, y, z = e3nn.o3.angles_to_xyz(alpha, beta)
    return x, y, z, alpha, beta


@cached_dirpklgz("cache/sh_grids")
def spherical_harmonics_on_grid(L, n):
    x, y, z, alpha, beta = spherical_surface(n)
    print(x.shape, alpha.shape)
    return x, y, z, e3nn.o3.spherical_harmonics(L, alpha, beta)
 

@cached_dirpklgz("cache/wigner_D_grids")
def wigner_D_on_grid(L, n):
    alpha, beta, gamma = euler_angles_on_grid(n)
    shape = alpha.shape
    abc = torch.stack([alpha.flatten(), beta.flatten(), gamma.flatten()],
                      dim=-1)
    wig_Ds = []
    for a, b, c in abc:
        wig_Ds.append(e3nn.o3.irr_repr(L, a, b, c))
    wig_D_shape = wig_Ds[0].shape
    return torch.stack(wig_Ds).reshape(shape + wig_D_shape)

