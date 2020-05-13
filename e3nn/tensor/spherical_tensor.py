# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import math

import torch

from e3nn import o3, rs, rsh
from e3nn.s2grid import ToS2Grid
from e3nn.tensor.irrep_tensor import IrrepTensor
from e3nn.tensor_product import TensorProduct


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


class SphericalTensor:
    def __init__(self, signal: torch.Tensor, lmax: int, p_val: int = 0, p_arg: int = 0):
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
        self.Rs = rs.convention([(1, l, p_val * p_arg**l) for l in range(lmax + 1)])
        self.p_val = p_val
        self.p_arg = p_arg

    @classmethod
    def from_geometry(cls, vectors, lmax, p=0):
        """
        :param vectors: tensor of vectors (p=-1) or pseudovectors (p=1)
        """
        signal = adjusted_projection(vectors, lmax)
        return cls(signal, lmax, p_val=1, p_arg=p)

    def sph_norm(self):
        i = 0
        norms = []
        for l in range(self.lmax + 1):
            n = self.signal[..., i: i + 2 * l + 1].norm(p=2, dim=-1)
            norms.append(n)
            i += 2 * l + 1
        return torch.stack(norms, dim=-1)

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
        new_Rs = [(1, l) for l in range(lmax + 1)]
        if self.lmax == lmax:
            return self
        elif self.lmax > lmax:
            new_signal = self.signal[..., :rs.dim(new_Rs)]
            return SphericalTensor(new_signal, lmax, self.p_val, self.p_arg)
        elif self.lmax < lmax:
            new_signal = torch.zeros(*self.signal.shape[:-1], rs.dim(new_Rs))
            new_signal[..., :rs.dim(self.Rs)] = self.signal
            return SphericalTensor(new_signal, lmax, self.p_val, self.p_arg)

    def __add__(self, other):
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)

        return SphericalTensor(new_self.signal + new_other.signal, new_self.lmax, self.p_val, self.p_arg)

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)

        mult = new_self.signal * new_other.signal
        mapping_matrix = rs.map_mul_to_Rs(new_self.Rs)
        scalars = torch.einsum('rm,...r->...m', mapping_matrix, mult)
        Rs = [(1, 0, p1 * p2) for (_, l1, p1), (_, l2, p2) in zip(new_self.Rs, new_other.Rs)]
        return IrrepTensor(scalars, Rs)

    def dot(self, other):
        return (self * other).tensor.sum(-1)

    def __matmul__(self, other):
        # Tensor product
        # Better handle mismatch of features indices
        tp = TensorProduct(self.Rs, other.Rs, o3.selection_rule)
        return IrrepTensor(tp(self.signal, other.signal), tp.Rs_out)
