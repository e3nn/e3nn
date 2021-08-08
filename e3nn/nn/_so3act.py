# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import o3
from e3nn.math import normalize2mom


class SO3Activation(torch.nn.Module):
    r'''Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    irreps : `o3.Irreps`
        input representation of the form ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    act : function
        activation function :math:`\phi`

    samples : int
        number of points to sample SO(3) (the higher the more accurate)

    normalization : {'norm', 'component'}

    lmax_out : int, optional
        maximum ``l`` of the output
    '''
    def __init__(self, irreps: o3.Irreps, act, samples, normalization='component', lmax_out=None):
        super().__init__()

        irreps = o3.Irreps(irreps).simplify()
        _, (_, p_val) = irreps[0]
        _, (lmax, _) = irreps[-1]
        assert all(mul == ir.dim for mul, ir in irreps)
        assert sorted(set(irreps.ls)) == list(range(lmax + 1))
        assert all(p == p_val for _, (l, p) in irreps), "the parity of the input is not well defined"
        self.irreps_in = irreps
        # the input transforms as : A_l ---> p_val * A_l
        # the signal transforms as : f(g) ---> p_val * f(g)
        if lmax_out is None:
            lmax_out = lmax
        assert lmax_out == lmax

        if p_val in (0, +1):
            self.irreps_out = o3.Irreps([(2 * l + 1, (l, p_val)) for l in range(lmax_out + 1)])
        if p_val == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = 1
                self.irreps_out = o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax_out + 1)])
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = -1
                self.irreps_out = o3.Irreps([(2 * l + 1, (l, -1)) for l in range(lmax_out + 1)])
            else:
                # p_act = 0
                raise ValueError("warning! the parity is violated")

        abc = o3.rand_angles(samples)
        assert normalization == "component"
        self.register_buffer(
            "D",
            torch.cat([
                (2 * l + 1)**0.5 * o3.wigner_D(l, *abc).transpose(1, 2).reshape(samples, (2 * l + 1)**2) for l in range(lmax + 1)
            ], dim=1)
        )
        self.act = normalize2mom(act)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        r'''evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor :math:`\{A^l\}_l` of shape ``(..., self.irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.irreps_out.dim)``
        '''
        assert features.shape[-1] == self.irreps_in.dim

        features = torch.einsum("...i,ni->...n", features, self.D) / self.D.shape[1]**0.5  # [..., random_points_on_SO3]
        features = self.act(features)
        features = torch.einsum("...n,ni->...i", features, self.D) * (self.D.shape[1]**0.5 / self.D.shape[0])

        return features
