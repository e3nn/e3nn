# pylint: disable=C,R,E1101
'''
Architecture to predict molecule energy on database qm7

RMSE test = 5.7
'''
import torch
import torch.nn as nn
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.blocks.tensor_product import TensorProductBlock
from util_cnn.model_backup import ModelBackup

import numpy as np
import pickle



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        features = [
            (5, 0, 0), # 64
            (10, 3, 0), # 32
            (10, 3, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 32
            (1, 0, 0) # 32
        ]
        common_block_params = {'size': 7, 'radial_amount': 3, 'padding': 3, 'batch_norm_momentum': 0.01}
        block_params = [
            {'stride': 2, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': True},
            {'stride': 1, 'relu': False},
        ]

        assert len(self.block_params) + 1 == len(self.features)

        blocks = [TensorProductBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]
        self.blocks = torch.nn.Sequential(*blocks)

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
        x = self.blocks(inp)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x*y*z]
        x = x.mean(-1) # [batch, features]

        x = x * self.alpha * 5

        inp = inp.view(inp.size(0), inp.size(1), -1).sum(-1)

        y = self.lin(inp)

        return x + y


#TODO replace this silly ModelBackup by Model with learning rate decay
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

    def number_of_epochs(self):
        return 100

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
