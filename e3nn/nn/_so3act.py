import torch
from e3nn import o3
from e3nn.math import normalize2mom
from e3nn.o3._s2grid import _quadrature_weights, s2_grid
from e3nn.util.jit import compile_mode


@compile_mode('script')
class SO3Activation(torch.nn.Module):
    r'''Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    irreps : `o3.Irreps`
        input representation of the form ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    act : function
        activation function :math:`\phi`

    resolution : int
        about ``resolution**3`` points are used to sample SO(3) (the higher the more accurate)

    normalization : {'norm', 'component'}

    lmax_out : int, optional
        maximum ``l`` of the output
    '''
    def __init__(self, irreps: o3.Irreps, act, resolution, normalization='component', lmax_out=None, aspect_ratio=2):
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

        assert normalization == "component"

        nb = 2 * resolution
        na = round(2 * aspect_ratio * resolution)

        b, a = s2_grid(nb, na)
        self.register_buffer(
            "D",
            torch.cat([
                (2 * l + 1)**0.5 * o3.wigner_D(
                    l, a[:, None, None], b[None, :, None], a[None, None, :]
                ).transpose(-1, -2).reshape(na, nb, na, (2 * l + 1)**2)
                for l in range(lmax + 1)
            ], dim=-1)
        )
        qw = _quadrature_weights(nb // 2) * nb**2 / na**2
        self.register_buffer('qw', qw)
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
        features = torch.einsum("...i,abci->...abc", features, self.D) / self.D.shape[-1]**0.5  # [..., random_points_on_SO3]
        features = self.act(features)
        features = torch.einsum("...abc,abci,b->...i", features, self.D, self.qw) * self.D.shape[-1]**0.5  # / self.D.shape[0])

        return features
