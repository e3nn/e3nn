# pylint: disable=C,R,E1101
'''
Minimalist example of usage of SE(3) CNN

This example train a neural network to classify two geometrical objects
A ball : x^2 + y^2 + z^2 < radius^2
An octahedron : |x| + |y| + |z| < radius

The input data is generated on the fly, the radius is random and noise is added
'''
import torch
import numpy as np

import torch.nn as nn

from se3cnn.non_linearities import NormRelu
from se3cnn.non_linearities import NormSoftplus
from se3cnn.non_linearities import ScalarActivation
from se3cnn import SO3

from se3cnn.batchnorm import SE3BatchNorm
from se3cnn.convolution import SE3Convolution
from se3cnn import SE3BNConvolution, SE3ConvolutionBN

from se3cnn.blocks import NormBlock

from se3cnn.util.optimizers_L1L2 import Adam
from se3cnn.util.lr_schedulers import lr_scheduler_exponential


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # The parameters of a GatedBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1, ),  # As input we have a scalar field
            (2, 2, 2, 2),  # Note that this particular choice of multiplicities it completely arbitrary
            (4, 4, 3, 3),
            (20, )  # Two scalar fields as output
        ]





        from se3cnn import basis_kernels
        radial_window_dict = {'radial_window_fct':basis_kernels.gaussian_window_fct_convenience_wrapper,
                              'radial_window_fct_kwargs':{'mode':'compromise', 'border_dist':0., 'sigma':.6}}
        common_block_params = {'size': 5, 'stride': 2, 'padding': 3, 'batch_norm_before_conv': False, 'radial_window_dict':radial_window_dict}






        block_params = [
            {'activation': torch.nn.functional.relu},  # the activation of the scalars
            {'activation': torch.nn.functional.relu},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)


        blocks = [NormBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(20, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2),
        )

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''

        x = self.sequence(inp)  # [batch, features]

        return x


def main():
    torch.backends.cudnn.benchmark = True

    model = CNN()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if torch.cuda.is_available():
        model.cuda()


    # split up parameters into groups, named_parameters() returns tupels ('name', parameter)
    # each group gets its own regularization gain
    convLayers      = [m for m in model.modules() if isinstance(m, (SE3Convolution, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))]
    normActivs      = [m for m in model.modules() if isinstance(m, (NormSoftplus, NormRelu))]
    batchnormLayers = [m for m in model.modules() if isinstance(m, (SE3BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    linearLayers    = [m for m in model.modules() if isinstance(m, nn.Linear)]
    weights_conv  = [p for m in convLayers      for n,p in m.named_parameters() if n.endswith('weight')]
    weights_bn    = [p for m in batchnormLayers for n,p in m.named_parameters() if n.endswith('weight')]
    weights_fully = [p for m in linearLayers    for n,p in m.named_parameters() if n.endswith('weight')] # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    biases_conv   = [p for m in convLayers      for n,p in m.named_parameters() if n.endswith('bias')]
    biases_activs = [p for m in normActivs      for n,p in m.named_parameters() if n.endswith('bias')]
    biases_bn     = [p for m in batchnormLayers for n,p in m.named_parameters() if n.endswith('bias')]
    biases_fully  = [p for m in linearLayers    for n,p in m.named_parameters() if n.endswith('bias')] # CROP OFF LAST WEIGHT !!!!! (classification layer)
    biases_fully, biases_softmax = biases_fully[:-1], [biases_fully[-1]]
    for np_tuple in model.named_parameters():
        if not np_tuple[0].endswith(('weight', 'weights_re', 'weights_im', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(np_tuple[0]))
    param_groups = [dict(params=weights_conv,    lamb_L1=0, lamb_L2=0),
                    dict(params=weights_bn,      lamb_L1=0, lamb_L2=0),
                    dict(params=weights_fully,   lamb_L1=0, lamb_L2=0),
                    dict(params=weights_softmax, lamb_L1=0, lamb_L2=0),
                    dict(params=biases_conv,     lamb_L1=0, lamb_L2=0),
                    dict(params=biases_activs,   lamb_L1=0, lamb_L2=0),
                    dict(params=biases_bn,       lamb_L1=0, lamb_L2=0),
                    dict(params=biases_fully,    lamb_L1=0, lamb_L2=0),
                    dict(params=biases_softmax,  lamb_L1=0, lamb_L2=0)]
    optimizer = Adam(param_groups, lr=1e-2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = Adam(model.parameters(), lr=1e-2)

    batch_size = 64
    sample_size = 24  # Size of the input cube

    mesh = np.linspace(-1, 1, sample_size)
    mx, my, mz = np.meshgrid(mesh, mesh, mesh)

    def step(i):
        # only noise
        x = 0.1 * np.random.randn(batch_size, 1, sample_size, sample_size, sample_size)
        # random labels: zero for the ball and one for the octahedron
        y = np.random.randint(0, 2, size=(batch_size,))

        # add the balls and octahedrons on top of the noise
        for j, label in enumerate(y):
            radius = 0.6 + np.random.rand() * (0.9 - 0.6)

            if label == 0:
                # ball
                mask = mx ** 2 + my ** 2 + mz ** 2 < radius ** 2 / np.pi ** (2 / 3)
            if label == 1:
                # octahedron
                mask = abs(mx) + abs(my) + abs(mz) < radius

            x[j, 0, mask] += np.random.randint(2) * 2 - 1

        # convert the input and labels into Pytorch tensor
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        # upload on the GPU if available
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        # forward and backward propagation
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        # download results on the CPU
        loss = loss.data.cpu().numpy()
        out = out.data.cpu().numpy()
        y = y.data.cpu().numpy()

        # compute the accuracy
        acc = np.sum(out.argmax(-1) == y) / batch_size

        # print("{}: acc={}% loss={}".format(i, 100 * acc, float(loss)))
        biases_means = [np.mean(b.data.cpu().numpy()) for b in biases_activs]
        print("{}: {:4} {:4} acc={}% loss={:4}".format(i, *biases_means, 100 * acc, float(loss)))

    for i in range(1000):
        step(i)

if __name__ == '__main__':
    main()
