# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import math

import numpy as np
# import scipy.signal
import torch

from e3nn import o3, rs, rsh
from e3nn.irrep_tensor import IrrepTensor
from e3nn.kernel_mod import FrozenKernel
from e3nn.s2grid import ToS2Grid


def spherical_harmonics_dirac(vectors, lmax):
    """
    approximation of a signal that is 0 everywhere except on the angle (alpha, beta) where it is one.
    the higher is lmax the better is the approximation
    """
    return 4 * math.pi / (lmax + 1)**2 * rsh.spherical_harmonics_xyz(list(range(lmax + 1)), vectors)


def projection(vectors, lmax):
    """
    :param vectors: tensor of shape [..., xyz]
    :return: tensor of shape [..., l * m]
    """
    coeff = spherical_harmonics_dirac(vectors, lmax)  # [..., l * m]
    radii = vectors.norm(2, -1)  # [...]
    coeff[radii == 0] = 0
    return coeff * radii.unsqueeze(-1)


def adjusted_projection(vectors, lmax):
    """
    :param vectors: tensor of shape [..., xyz]
    :return: tensor of shape [l * m]
    """
    vectors = vectors.reshape(-1, 3)
    radii = vectors.norm(2, -1)  # [batch]
    vectors = vectors[radii > 0]  # [batch, 3]

    coeff = projection(vectors, lmax)  # [batch, l * m]
    A = torch.einsum(
        "ai,bi->ab", rsh.spherical_harmonics_xyz(list(range(lmax + 1)), vectors), coeff)
    coeff *= torch.lstsq(radii, A).solution.reshape(-1).unsqueeze(-1)
    return coeff.sum(0)


class SphericalTensor():
    def __init__(self, signal, lmax, p_val=0, p_arg=0):
        """
        f: s2 -> R

        Rotations
        [D(g) f](x) = f(g^{-1} x)

        Parity
        [P f](x) = p_val f(p_arg x)

        f(x) = sum F^l . Y^l(x)

        This class contains the F^l

        Rotations
        [D(g) f](x) = sum [D^l(g) F^l] . Y^l(x)         (using equiv. of Y and orthogonality of D)

        Parity
        [P f](x) = sum [p_val p_arg^l F^l] . Y^l(x)     (using parity of Y)
        """
        if signal.shape[-1] != (lmax + 1)**2:
            raise ValueError(
                "Last tensor dimension and Rs do not have same dimension.")

        self.signal = signal
        self.lmax = lmax
        self.mul = 1
        self.Rs = rs.convention([(self.mul, l, p_val * p_arg**l)
                                 for l in range(lmax + 1)])
        self.radial_model = None

    @classmethod
    def from_geometry(cls, vectors, lmax, p=0):
        """
        :param vectors: tensor of vectors (p=-1) or pseudovectors (p=1)
        """
        signal = adjusted_projection(vectors, lmax)
        return cls(signal, lmax, p_val=1, p_arg=p)

    def sph_norm(self):
        Rs = self.Rs
        signal = self.signal
        n_mul = sum([mul for mul, l, p in Rs])
        # Keep shape after Rs the same
        norms = torch.zeros(n_mul, *signal.shape[1:])
        sig_index = 0
        norm_index = 0
        for mul, l, _p in Rs:
            for _ in range(mul):
                norms[norm_index] = signal[sig_index: sig_index +
                                           (2 * l + 1)].norm(2, 0)
                norm_index += 1
                sig_index += 2 * l + 1
        return norms

    def signal_xyz(self, r):
        """
        Evaluate the signal on the sphere
        """
        sh = rsh.spherical_harmonics_xyz(list(range(self.lmax + 1)), r)
        dim = (self.lmax + 1)**2
        output = torch.einsum(
            'ai,zi->za', sh.reshape(-1, dim), self.signal.reshape(-1, dim))
        return output.reshape((*self.signal.shape[:-1], *r.shape[:-1]))

    def signal_alpha_beta(self, alpha, beta):
        """
        Evaluate the signal on the sphere
        """
        sh = rsh.spherical_harmonics_alpha_beta(
            list(range(self.lmax + 1)), alpha, beta)
        dim = (self.lmax + 1)**2
        output = torch.einsum(
            'ai,zi->za', sh.reshape(-1, dim), self.signal.reshape(-1, dim))
        return output.reshape((*self.signal.shape[:-1], *alpha.shape))

    def signal_on_grid(self, n=100):
        """
        Evaluate the signal on the sphere
        """
        grid = ToS2Grid(self.lmax, res=n, normalization='none')
        beta, alpha = torch.meshgrid(grid.betas, grid.alphas)  # [beta, alpha]
        r = o3.angles_to_xyz(alpha, beta)  # [beta, alpha, 3]
        return r, grid(self.signal)

    def plot(self, n=100, radius=True, center=None, relu=True):
        """
        r, f = self.plot()
        surface = go.Surface(
            x=r[:, :, 0].numpy(),
            y=r[:, :, 1].numpy(),
            z=r[:, :, 2].numpy(),
            surfacecolor=f.numpy()
        )
        fig = go.Figure(data=[surface])
        fig.show()
        """
        r, f = self.signal_on_grid(n)
        f = f.relu() if relu else f

        r = torch.cat([r, r[:, :1]], dim=1)  # [beta, alpha, 3]
        f = torch.cat([f, f[:, :1]], dim=1)  # [beta, alpha]

        if radius:
            r *= f.abs().unsqueeze(-1)

        if center is not None:
            r += center.unsqueeze(0).unsqueeze(0)

        return r, f

    def change_lmax(self, lmax):
        new_Rs = [(self.mul, l) for l in range(lmax + 1)]
        if self.lmax == lmax:
            return self
        elif self.lmax > lmax:
            new_signal = self.signal[:rs.dim(new_Rs)]
            return SphericalTensor(new_signal, lmax)
        elif self.lmax < lmax:
            new_signal = torch.zeros(rs.dim(new_Rs))
            new_signal[:rs.dim(self.Rs)] = self.signal
            return SphericalTensor(new_signal, lmax)

    def __add__(self, other):
        if self.mul != other.mul:
            raise ValueError("Multiplicities do not match.")
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)
        return SphericalTensor(new_self.signal + new_other.signal, self.lmax)

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        if self.mul != other.mul:
            raise ValueError("Multiplicities do not match.")
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)

        mult = (new_self.signal * new_other.signal)
        mapping_matrix = rs.map_mul_to_Rs(new_self.Rs)
        scalars = torch.einsum('rm,r->m', mapping_matrix, mult)
        return IrrepTensor(scalars, [(new_self.mul * (new_self.lmax + 1), 0)])

    def dot(self, other):
        scalars = self.__mul__(other)
        dot = scalars.tensor.sum(-1)
        dot /= (self.signal.norm(2, 0) * other.signal.norm(2, 0))
        return dot

    def __matmul__(self, other):
        # Tensor product
        # Better handle mismatch of features indices
        Rs_out, C = rs.tensor_product(self.Rs, other.Rs, o3.selection_rule)
        new_signal = torch.einsum('kij,...i,...j->...k',
                                  (C, self.signal, other.signal))
        return IrrepTensor(new_signal, Rs_out)

    def __rmatmul__(self, other):
        # Tensor product
        return self.__matmul__(other)


class FourierTensor():
    def __init__(self, signal, mul, lmax, p_val=0, p_arg=0):
        """
        f: s2 x r -> R^N

        Rotations
        [D(g) f](x) = f(g^{-1} x)

        Parity
        [P f](x) = p_val f(p_arg x)

        f(x) = sum F^l . Y^l(x)

        This class contains the F^l

        Rotations
        [D(g) f](x) = sum [D^l(g) F^l] . Y^l(x)         (using equiv. of Y and orthogonality of D)

        Parity
        [P f](x) = sum [p_val p_arg^l F^l] . Y^l(x)     (using parity of Y)
        """
        if signal.shape[-1] != mul * (lmax + 1)**2:
            raise ValueError(
                "Last tensor dimension and Rs do not have same dimension.")

        self.signal = signal
        self.lmax = lmax
        self.mul = mul
        self.Rs = rs.convention([(mul, l, p_val * p_arg**l)
                                 for l in range(lmax + 1)])
        self.radial_model = None

    @classmethod
    def from_geometry_with_radial(cls, vectors, radial_model, lmax, sum_points=True):
        """
        :param vectors: tensor of shape [..., xyz]
        :param radial_model: function of signature R+ -> R^mul
        :param lmax: maximal order of the signal
        """
        size = vectors.shape[:-1]
        vectors = vectors.reshape(-1, 3)  # [N, 3]
        radii = vectors.norm(2, -1)
        radial_functions = radial_model(radii)
        *_size, R = radial_functions.shape
        Rs = [(R, L) for L in range(lmax + 1)]
        mul_map = rs.map_mul_to_Rs(Rs)
        radial_functions = torch.einsum('nr,dr->nd',
                                        radial_functions.repeat(1, lmax + 1),
                                        mul_map)  # [N, signal]

        Ys = projection(vectors / radii.unsqueeze(-1), lmax)  # [N, l * m]
        irrep_map = rs.map_irrep_to_Rs(Rs)
        Ys = torch.einsum('nc,dc->nd', Ys, irrep_map)  # [N, l * mul * m]

        signal = Ys * radial_functions  # [N, l * mul * m]

        if sum_points:
            signal = signal.sum(0)
        else:
            signal = signal.reshape(*size, -1)

        new_cls = cls(signal, R, lmax)
        new_cls.radial_model = radial_model
        return new_cls

    def sph_norm(self):
        Rs = self.Rs
        signal = self.signal
        n_mul = sum([mul for mul, l, p in Rs])
        # Keep shape after Rs the same
        norms = torch.zeros(n_mul, *signal.shape[1:])
        sig_index = 0
        norm_index = 0
        for mul, l, _p in Rs:
            for _ in range(mul):
                norms[norm_index] = signal[sig_index: sig_index +
                                           (2 * l + 1)].norm(2, 0)
                norm_index += 1
                sig_index += 2 * l + 1
        return norms

    def plot(self, box_length, center=None,
             sh=rsh.spherical_harmonics_xyz, n=30,
             radial_model=None, relu=True):
        muls, _ls, _ps = zip(*self.Rs)
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

        if center is not None:
            r += center.unsqueeze(0)

        return r, f

    def change_lmax(self, lmax):
        new_Rs = [(self.mul, l) for l in range(lmax + 1)]
        if self.lmax == lmax:
            return self
        elif self.lmax > lmax:
            new_signal = self.signal[:rs.dim(new_Rs)]
            return FourierTensor(new_signal, self.mul, lmax)
        elif self.lmax < lmax:
            new_signal = torch.zeros(rs.dim(new_Rs))
            new_signal[:rs.dim(self.Rs)] = self.signal
            return FourierTensor(new_signal, self.mul, lmax)

    def __add__(self, other):
        if self.mul != other.mul:
            raise ValueError("Multiplicities do not match.")
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)
        return FourierTensor(new_self.signal + new_other.signal, self.mul, self.lmax)

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        if self.mul != other.mul:
            raise ValueError("Multiplicities do not match.")
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)

        mult = (new_self.signal * new_other.signal)
        mapping_matrix = rs.map_mul_to_Rs(new_self.Rs)
        scalars = torch.einsum('rm,r->m', mapping_matrix, mult)
        return IrrepTensor(scalars, [(new_self.mul * (new_self.lmax + 1),0)])

    def dot(self, other):
        scalars = self.__mul__(other)
        dot = scalars.tensor.sum(-1)
        dot /= (self.signal.norm(2, 0) * other.signal.norm(2, 0))
        return dot

    def __matmul__(self, other):
        # Tensor product
        # Better handle mismatch of features indices
        Rs_out, C = rs.tensor_product(self.Rs, other.Rs, o3.selection_rule)
        new_signal = torch.einsum('kij,...i,...j->...k',
                                  (C, self.signal, other.signal))
        return IrrepTensor(new_signal, Rs_out)

    def __rmatmul__(self, other):
        # Tensor product
        return self.__matmul__(other)


def plot_on_grid(box_length, radial_model, Rs, sh=rsh.spherical_harmonics_xyz, n=30):
    l_to_index = {}
    set_of_l = set([l for mul, l, p in Rs])
    start = 0
    for l in set_of_l:
        l_to_index[l] = [start, start + 2 * l + 1]
        start += 2 * l + 1

    r = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j].reshape(3, -1)
    r = r.transpose(1, 0)
    r *= box_length / 2.
    r = torch.from_numpy(r)

    Rs_in = [(1, 0)]
    Rs_out = Rs

    def radial_lambda(_ignored):
        return radial_model

    grid = FrozenKernel(Rs_in, Rs_out, radial_lambda, r, sh=sh)
    R = grid.R(grid.radii)
    # j is just 1 because Rs_in is 1d
    f = torch.einsum('xjmw,xw->xj', grid.Q, R)
    return r, f
