from functools import partial

import torch

from e3nn.radial import CosineBasisModel
from e3nn.non_linearities.rescaled_act import ShiftedSoftplus
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.point.kernelconv import KernelConv, KernelQM9
from e3nn.rs import dim

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

geometry = torch.rand(100, 40, 3, dtype=torch.float32).to(device)
features = torch.rand(100, 40, 2, dtype=torch.float32).to(device)


def main():
    print("name: reserved allocated")
    ssp = ShiftedSoftplus(5.0)
    radial = partial(CosineBasisModel, max_radius=10., number_of_basis=25, h=100, L=3, act=ssp)
    # kernel_conv = KernelConv([(2, 0)], [(5, 0), (5, 1)], RadialModel=radial)
    # f2 = kernel_conv(features, geometry)
    K9 = partial(KernelQM9, RadialModel=radial)
    C9 = Convolution(K9, [(2, 0)], [(5, 0), (5, 1)])
    f2 = C9(features, geometry)
    loss = torch.norm(f2 - torch.zeros_like(f2))
    loss.backward()
    print('done')


if __name__ == '__main__':
    main()
