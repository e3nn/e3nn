# pylint: disable=not-callable, no-member, invalid-name, line-too-long, missing-docstring, arguments-differ
import numpy as np
import torch

from e3nn import rs
from e3nn.kernel_mod import FrozenKernel
from e3nn.tensor.spherical_tensor import projection


class FourierTensor:
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
    def from_geometry(cls, vectors, radial_model, lmax, sum_points=True):
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

    def plotly_surface(self, box_length, center=None, n=30,
                       radial_model=None, relu=True):
        """
        To use as follow
        ```
        import plotly.graph_objects as go
        surface = go.Surface(**self.plotly_surface())
        fig = go.Figure(data=[surface])
        fig.show()
        ```
        """
        r, f = self.plot(box_length, center, n, radial_model, relu)
        return dict(
            x=r[:, :, 0].numpy(),
            y=r[:, :, 1].numpy(),
            z=r[:, :, 2].numpy(),
            surfacecolor=f.numpy(),
        )

    def plot(self, box_length, center=None, n=30,
             radial_model=None, relu=True):
        muls, _ls, _ps = zip(*self.Rs)
        # We assume radial functions are repeated across L's
        assert len(set(muls)) == 1
        num_L = len(self.Rs)
        if radial_model is None:
            radial_model = self.radial_model

        def new_radial(x):
            return radial_model(x).repeat(1, num_L)  # Repeat along filter dim
        r, f = plot_on_grid(box_length, new_radial, self.Rs, n=n)
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


def plot_on_grid(box_length, radial_model, Rs, n=30):
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

    grid = FrozenKernel(Rs_in, Rs_out, radial_lambda, r)
    f = grid()
    f = f[..., 0]
    return r, f
