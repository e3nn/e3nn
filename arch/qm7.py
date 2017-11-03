# pylint: disable=C,R,E1101
'''
Architecture to predict molecule energy on database qm7

RMSE test = 5.7
'''
import torch
import torch.nn as nn
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.bn_conv import SE3BNConvolution
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn.non_linearities.tensor_product import TensorProduct
from se3_cnn import SO3
from util_cnn.model_backup import ModelBackup

from util_cnn import time_logging
import logging
import numpy as np
import pickle

logger = logging.getLogger("trainer")


class Block(nn.Module):
    def __init__(self, repr_in, repr_out, relu, stride):
        super().__init__()
        self.tensor = TensorProduct([(repr_in[0], 1, False), (repr_in[1], 3, True), (repr_in[2], 5, False)]) if repr_in[1] > 0 else None
        self.bn_conv = SE3BNConvolution(
            size=7,
            radial_amount=3,
            Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5), (repr_in[1], SO3.repr3x3)],
            Rs_out=[(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)],
            stride=stride,
            padding=3,
            momentum=0.01,
            mode='maximum')
        self.relu = BiasRelu([(repr_out[0], True), (repr_out[1] * 3, False), (repr_out[2] * 5, False)], normalize=False) if relu else None

    def forward(self, sv5): # pylint: disable=W
        if self.tensor is not None:
            t = self.tensor(sv5)
            sv5t = torch.cat([sv5, t], dim=1)
        else:
            sv5t = sv5

        sv5 = self.bn_conv(sv5t)

        if self.relu is not None:
            sv5 = self.relu(sv5)
        return sv5

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.features = [
            (5, 0, 0), # 64
            (10, 3, 0), # 32
            (10, 3, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 32
            (1, 0, 0) # 32
        ]
        self.block_params = [
            {'stride': 2, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': False},
        ]

        assert len(self.block_params) + 1 == len(self.features)

        for i in range(len(self.block_params)):
            block = Block(self.features[i], self.features[i + 1], self.block_params[i]['relu'], self.block_params[i]['stride'])
            setattr(self, 'block{}'.format(i), block)

        self.lin = torch.nn.Linear(5, 1)
        self.lin.weight.data[0, 0] = -69.14
        self.lin.weight.data[0, 1] = -153.3
        self.lin.weight.data[0, 2] = -99.04
        self.lin.weight.data[0, 3] = -97.76
        self.lin.weight.data[0, 4] = -80.44

        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, inp): # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = inp

        t = time_logging.start()
        for i in range(len(self.block_params)):
            block = getattr(self, 'block{}'.format(i))
            x = block(x)
            t = time_logging.end("block {}".format(i), t)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x*y*z]
        x = x.mean(-1) # [batch, features]

        x = x * self.alpha * 5

        inp = inp.view(inp.size(0), inp.size(1), -1).sum(-1)

        y = self.lin(inp)

        # print(repr(x.data.cpu().numpy()), repr(y.data.cpu().numpy()))

        return x + y


class MyModel(ModelBackup):
    def __init__(self):
        super().__init__(
            success_factor=1,
            decay_factor=2 ** (-1/4 * 1/6),
            reject_factor=2 ** (-1),
            reject_ratio=2,
            min_learning_rate=1e-4,
            max_learning_rate=0.2,
            initial_learning_rate=2e-3)

    def initialize(self, **kargs):
        self.cnn = CNN()
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def get_batch_size(self, epoch=None):
        return 16

    def get_criterion(self):
        return torch.nn.MSELoss()

    def get_learning_rate(self, epoch):
        for module in self.cnn.modules():
            if isinstance(module, SE3BatchNorm):
                module.momentum = 0.01 * (0.1 ** epoch)

        return super().get_learning_rate(epoch)

    def load_files(self, files):
        p = 0.3
        n = 64

        number_of_atoms_types = 5
        inputs = np.zeros((len(files), number_of_atoms_types, n, n, n), dtype=np.float32)

        a = np.linspace(start=-n/2*p + p/2, stop=n/2*p - p/2, num=n, endpoint=True)
        xx, yy, zz = np.meshgrid(a, a, a, indexing="ij")

        for i, f in enumerate(files):
            with open(f, 'rb') as f:
                content = pickle.load(f)

            for ato, pos in zip(content[0], content[1]):
                x = pos[0]
                y = pos[1]
                z = pos[2]

                density = np.exp(-((xx-x)**2 + (yy-y)**2 + (zz-z)**2) / (2 * p**2))
                density /= np.sum(density)

                inputs[i, ato] += density

        return inputs
