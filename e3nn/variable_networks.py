import torch
from functools import partial

#from e3nn.networks import *
from e3nn import o3, rs
from e3nn.kernel import Kernel
from e3nn.non_linearities import GatedBlock, GatedBlockParity
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.point.operations import Convolution
from e3nn.radial import GaussianRadialModel
from e3nn.tensor_product import LearnableTensorSquare
from e3nn.batchnorm import BatchNorm

class VariableParityNetwork(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, lmaxes, muls, 
                 max_radius=1.0, number_of_basis=3, radial_layers=10, radial_h=100,
                 feature_product=False, kernel=Kernel, convolution=Convolution,
                 radial_model=None, batch_norm=True):
        """
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param lmaxes: list of integers, giving the lmax for each layer (and, implicitly,
                        the number of layers)
        :param muls: either an integer (in which case, same multiplicity for all layers and l's)
                     or a list of integers (in which case, different multiplicity for each layer)
                     or a list of lists of integers (in which case, specify multiplicity per each layer and l)
        :param max_radius: float, how far the radial basis extends
        :param number_of_basis: number of radial basis functions
        :param radial_layers: number of layers in radial function network
        :param radial_h: nodes per layer in radial function network
        :param feature_product, kernel, convolution, radial_model: ...
        :param batch_norm: boolean, whether to put batch norm between layers

        """
        super().__init__()

        self.batch_norm = batch_norm
        if radial_model is None:
            radial_model = partial(GaussianRadialModel, number_of_basis=number_of_basis)
        R = partial(radial_model, max_radius=max_radius, h=radial_h,
                    L=radial_layers, act=swish)

        modules = []

        Rs = Rs_in
        if isinstance(muls, int): muls = [muls] * len(lmaxes)

        for mul, lmax in zip(muls, lmaxes):
            if isinstance(mul, int): mul = [mul] * (lmax + 1)
            scalars = [(m, l, p) for m, l, p in [(mul[0], 0, +1), (mul[0], 0, -1)] if rs.haslinearpath(Rs, l, p)]
            act_scalars = [(m, swish if p == 1 else tanh) for m, l, p in scalars]

            nonscalars = [(m, l, p) for (l,m) in enumerate(mul) for p in [+1, -1] if rs.haslinearpath(Rs, l, p)]
            gates = [(rs.mul_dim(nonscalars), 0, +1)]
            act_gates = [(-1, sigmoid)]

            act = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
            conv = convolution(kernel(Rs, act.Rs_in, RadialModel=R, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax)))
            if batch_norm:
                bn = BatchNorm([(m,2 * l + 1) for (m,l,_) in act.Rs_in])

            if feature_product:
                tr1 = rs.TransposeToMulL(act.Rs_out)
                lts = LearnableTensorSquare(tr1.Rs_out, [(1, l, p) for l in range(lmax + 1) for p in [-1, 1]], allow_change_output=True)
                tr2 = torch.nn.Flatten(2)
                act = torch.nn.Sequential(act, tr1, lts, tr2)
                Rs = tr1.mul * lts.Rs_out
            else:
                Rs = act.Rs_out

            if batch_norm:
                block = torch.nn.ModuleList([conv, bn, act])
            else:
                block = torch.nn.ModuleList([conv, act])
            modules.append(block)

        self.layers = torch.nn.ModuleList(modules)
    
        self.layers.append(convolution(kernel(Rs, Rs_out, RadialModel=R,
                selection_rule=partial(o3.selection_rule_in_out_sh,
                lmax=lmaxes[-1]), allow_unused_inputs=True)))

        self.feature_product = feature_product

    def forward(self, input, *args, **kwargs):
        output = input
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = args[0].shape[-2]

        for layer in self.layers:
            if isinstance(layer, torch.nn.ModuleList):
                for i, sublayer in enumerate(layer):
                    if i == 0:
                        output = sublayer(output, *args, **kwargs)
                    else:
                        output = sublayer(output)
            else:
                output = layer(output, *args, **kwargs)
        
        return output

