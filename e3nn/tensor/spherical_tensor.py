# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import math

import torch

from e3nn import o3, rs, rsh
from e3nn.s2grid import ToS2Grid, FromS2Grid
from e3nn.tensor.irrep_tensor import IrrepTensor


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
    radii = vectors.norm(2, dim=-1, keepdim=True)  # [...]
    return coeff * radii


def adjusted_projection(vectors, lmax):
    """
    :param vectors: tensor of shape [..., xyz]
    :return: tensor of shape [l * m]
    """
    vectors = vectors.reshape(-1, 3)
    radii = vectors.norm(2, -1)  # [batch]
    vectors = vectors[radii > 0]  # [batch, 3]

    coeff = rsh.spherical_harmonics_xyz(list(range(lmax + 1)), vectors)  # [batch, l * m]
    A = torch.einsum(
        "ai,bi->ab",
        coeff,
        coeff
    )
    # Y(v_a) . Y(v_b) solution_b = radii_a
    solution = torch.lstsq(radii, A).solution.reshape(-1)  # [b]
    assert (radii - A @ solution).abs().max() < 1e-5 * radii.abs().max()

    return solution @ coeff


class SphericalTensor:
    def __init__(self, signal: torch.Tensor, p_val: int = 0, p_arg: int = 0):
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
        lmax = round(math.sqrt(signal.shape[-1]) - 1)

        if signal.shape[-1] != (lmax + 1)**2:
            raise ValueError(
                "Last tensor dimension and Rs do not have same dimension.")

        self.signal = signal
        self.lmax = lmax
        self.Rs = rs.convention([(1, l, p_val * p_arg**l) for l in range(lmax + 1)])
        self.p_val = p_val
        self.p_arg = p_arg

    @classmethod
    def from_geometry(cls, vectors, lmax, p=0, adjusted=True):
        """
        :param vectors: tensor of vectors (p=-1) or pseudovectors (p=1) of shape [..., 3=xyz]
        """
        if adjusted:
            signal = adjusted_projection(vectors, lmax)
        else:
            vectors = vectors.reshape(-1, 3)
            r = vectors.norm(dim=1)
            sh = rsh.spherical_harmonics_xyz(list(range(lmax + 1)), vectors)
            # 0.5 * sum_a ( Y(v_a) . sum_b r_b Y(v_b) s - r_a )^2
            A = torch.einsum('ai,b,bi->a', sh, r, sh)
            # 0.5 * sum_a ( A_a s - r_a )^2
            # sum_a A_a^2 s = sum_a A_a r_a
            s = torch.dot(A, r) / A.norm().pow(2)
            signal = s * torch.einsum('a,ai->i', r, sh)
        return cls(signal, p_val=1, p_arg=p)

    @classmethod
    def from_samples(cls, positions, values, lmax, res=100, p_val=0, p_arg=0):
        """
        :param positions: tensor of shape [num_points, 3=xyz]
        :param values: tensor of shape [num_points]
        """
        positions = positions.reshape(-1, 3)
        values = values.reshape(-1)
        positions /= positions.norm(p=2, dim=1, keepdim=True)
        assert positions.shape[0] == values.shape[0], "positions and values must have the same number of points"

        s2 = FromS2Grid(res=res, lmax=lmax, normalization='none')
        pos = s2.grid

        cd = torch.cdist(pos, positions, p=2)
        val = values[cd.argmin(2)]

        return cls(s2(val), p_val=p_val, p_arg=p_arg)

    @classmethod
    def spherical_harmonic(cls, l, m, lmax=None):
        if lmax is None:
            lmax = l
        signal = torch.zeros((lmax + 1)**2)
        signal[l**2 + l + m] = 1
        return cls(signal)

    def __repr__(self):
        p_str = ""
        if self.p_arg != 0 and self.p_val != 0:
            p_str = f", [Parity f](x) = {'-' if self.p_val == -1 else ''}f({'-' if self.p_arg == -1 else ''}x)"
        return f"{self.__class__.__name__}(lmax={self.lmax}{p_str})"

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

    def signal_on_grid(self, res=100):
        """
        :return: [..., beta, alpha]
        Evaluate the signal on the sphere
        """
        s2 = ToS2Grid(self.lmax, res=res, normalization='none')
        return s2.grid, s2(self.signal)

    def plotly_surface(self, res=100, radius=True, center=None, relu=False):
        """
        To use as follow
        ```
        import plotly.graph_objects as go
        surface = go.Surface(**self.plotly_surface())
        fig = go.Figure(data=[surface])
        fig.show()
        ```
        """
        r, f = self.plot(res, radius, center, relu)
        return dict(
            x=r[:, :, 0].numpy(),
            y=r[:, :, 1].numpy(),
            z=r[:, :, 2].numpy(),
            surfacecolor=f.numpy(),
        )

    def plot(self, res=100, radius=True, center=None, relu=False):
        """
        Returns `(r, f)` of shapes `[beta, alpha, 3]` and `[beta, alpha]`
        """
        assert self.signal.dim() == 1

        r, f = self.signal_on_grid(res)
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

    def change_lmax(self, lmax):
        new_Rs = [(1, l) for l in range(lmax + 1)]
        if self.lmax == lmax:
            return self
        elif self.lmax > lmax:
            new_signal = self.signal[..., :rs.dim(new_Rs)]
            return SphericalTensor(new_signal, self.p_val, self.p_arg)
        elif self.lmax < lmax:
            new_signal = torch.zeros(*self.signal.shape[:-1], rs.dim(new_Rs))
            new_signal[..., :rs.dim(self.Rs)] = self.signal
            return SphericalTensor(new_signal, self.p_val, self.p_arg)

    def __add__(self, other):
        lmax = max(self.lmax, other.lmax)
        new_self = self.change_lmax(lmax)
        new_other = other.change_lmax(lmax)

        return SphericalTensor(new_self.signal + new_other.signal, self.p_val, self.p_arg)

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
        tp = rs.TensorProduct(self.Rs, other.Rs, o3.selection_rule)
        return IrrepTensor(tp(self.signal, other.signal), tp.Rs_out)

    @classmethod
    def from_irrep_tensor(cls, irrep_tensor):
        Rs_remove_p = [(mul, L) for mul, L, p in irrep_tensor.Rs]
        Rs, perm = rs.sort(Rs_remove_p)
        Rs = rs.simplify(Rs)
        mul, Ls, p  = zip(*Rs)
        if max(mul) > 1:
            raise ValueError(
                "Cannot have multiplicity greater than 1 for any L. This tensor has a simplified Rs of {}".format(Rs)
            )
        Lmax = max(Ls)
        sorted_tensor = torch.einsum('ij,...j->...i', perm.to_dense(), irrep_tensor.tensor)
        signal = torch.zeros((Lmax + 1)**2)
        Rs_idx = 0
        for L in range(Lmax + 1):
            if Rs[Rs_idx][1] == L:
                ten_slice = slice(rs.dim(Rs[:Rs_idx]), rs.dim(Rs[:Rs_idx + 1]))
                signal[L ** 2: (L + 1) ** 2] = sorted_tensor[ten_slice]
                Rs_idx += 1
        return cls(signal)
