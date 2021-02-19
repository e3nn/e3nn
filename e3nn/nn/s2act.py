# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import o3
from e3nn.math import normalize2mom


class S2Activation(torch.nn.Module):
    r'''Apply non linearity on the signal on the sphere

    | Maps to the sphere, apply the non linearity point wise and project back.
    | The signal on the sphere is a quasiregular representation of :math:`O(3)` and we can apply a pointwise operation on these representations.

    .. math:: \{A^l\}_l \mapsto \{\int \phi(\sum_l A^l \cdot Y^l(x)) Y^j(x) dx\}_j

    Parameters
    ----------
    irreps : `Irreps`
        input representation of the form ``[(1, (l, p_val * (p_arg)^l)) for l in [0, ..., lmax]]``

    act : function
        activation function :math:`\phi`

    res : int
        resolution of the grid on the sphere (the higher the more accurate)

    normalization : {'norm', 'component'}

    lmax_out : int, optional
        maximum ``l`` of the output

    random_rot : bool
        rotate randomly the grid

    Examples
    --------
    >>> from e3nn import io
    >>> m = S2Activation(io.SphericalTensor(5, p_val=+1, p_arg=-1), torch.tanh, 100)
    '''
    def __init__(self, irreps, act, res, normalization='component', lmax_out=None, random_rot=False):
        super().__init__()

        irreps = o3.Irreps(irreps).simplify()
        _, (_, p_val) = irreps[0]
        _, (lmax, _) = irreps[-1]
        assert all(mul == 1 for mul, _ in irreps)
        assert irreps.ls == [l for l in range(lmax + 1)]
        if all(p == p_val for _, (l, p) in irreps):
            p_arg = 1
        elif all(p == p_val * (-1) ** l for _, (l, p) in irreps):
            p_arg = -1
        else:
            assert False, "the parity of the input is not well defined"
        self.irreps_in = irreps
        # the input transforms as : A_l ---> p_val * (p_arg)^l * A_l
        # the sphere signal transforms as : f(r) ---> p_val * f(p_arg * r)
        if lmax_out is None:
            lmax_out = lmax

        if p_val == +1 or p_val == 0:
            self.irreps_out = o3.Irreps([(1, (l, p_val * p_arg ** l)) for l in range(lmax_out + 1)])
        if p_val == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = 1
                self.irreps_out = o3.Irreps([(1, (l, p_arg ** l)) for l in range(lmax_out + 1)])
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                # p_act = -1
                self.irreps_out = o3.Irreps([(1, (l, -p_arg ** l)) for l in range(lmax_out + 1)])
            else:
                # p_act = 0
                raise ValueError("warning! the parity is violated")

        self.to_s2 = o3.ToS2Grid(lmax, res, normalization=normalization)
        self.from_s2 = o3.FromS2Grid(res, lmax_out, normalization=normalization, lmax_in=lmax)
        self.act = normalize2mom(act)
        self.random_rot = random_rot

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

        if self.random_rot:
            abc = o3.rand_angles(dtype=features.dtype, device=features.device)
            features = torch.einsum('ij,...j->...i', self.irreps_in.D_from_angles(*abc), features)

        features = self.to_s2(features)  # [..., beta, alpha]
        features = self.act(features)
        features = self.from_s2(features)

        if self.random_rot:
            features = torch.einsum('ij,...j->...i', self.irreps_out.D_from_angles(*abc).T, features)
        return features
