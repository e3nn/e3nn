import torch
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode
from e3nn.o3 import SO3Grid


@compile_mode('script')
class SO3Activation(torch.nn.Module):
    r'''Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax_in : int
        input lmax

    lmax_out : int
        output lmax

    act : function
        activation function :math:`\phi`

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}
    '''
    def __init__(self, lmax_in, lmax_out, act, resolution, *, normalization='component', aspect_ratio=2):
        super().__init__()

        self.grid_in = SO3Grid(lmax_in, resolution, normalization=normalization, aspect_ratio=aspect_ratio)
        self.grid_out = SO3Grid(lmax_out, resolution, normalization=normalization, aspect_ratio=aspect_ratio)
        self.act = normalize2mom(act)

        self.lmax_in = lmax_in
        self.lmax_out = lmax_out

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.lmax_in} -> {self.lmax_out})"

    def forward(self, features):
        r'''evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.irreps_out.dim)``
        '''
        features = self.grid_in.to_grid(features)
        features = self.act(features)
        features = self.grid_out.from_grid(features)

        return features
