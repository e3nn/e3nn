# pylint: disable=C,R,E1101
'''
Based on c2

+ tensor product
'''
import torch
import torch.nn as nn
from se3_cnn.convolution import SE3Convolution
from se3_cnn.non_linearities.tensor_product import TensorProduct
from se3_cnn import SO3
from se3_cnn.train.model import Model
import logging
import numpy as np

logger = logging.getLogger("trainer")


class Block(nn.Module):
    def __init__(self, scalar_out, vector, scalar_in, relu):
        super().__init__()
        self.conv1 = SE3Convolution(5, [(vector, SO3.repr3)], [(scalar_in, SO3.repr1)],
            bias_relu=False,
            norm_relu=False,
            scalar_batch_norm=False,
            radial_type="hat", # n = size
            stride=2,
            padding=3)
        self.tensor = TensorProduct([(vector, 3, True)])
        self.conv2 = SE3Convolution(5, [(scalar_out, SO3.repr1)], [(vector, SO3.repr3), (vector, SO3.repr3x3)],
            bias_relu=relu,
            norm_relu=False,
            scalar_batch_norm=True,
            radial_type="hat", # n = size
            stride=1,
            padding=3)

    def forward(self, s): # pylint: disable=W
        v = self.conv1(s)
        t = self.tensor(v)
        vt = torch.cat([v, t], dim=1)
        s = self.conv2(vt) # BN, bias and relu
        return s

class CNN(nn.Module):

    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        logger.info("Create CNN for classify %d classes", number_of_classes)

        features = [1, # 64
            8, # (64 + 2*3 - 4) / 2 + 2*3 - 4 = (64 + 2) / 2 + 2 = 35
            16, # (35 + 2)/2+2 = 20
            16, # (20+2)/2+2 = 13
            16, # (13+2)/2+2 = 9
            number_of_classes]  # (9+2)/2+2 = 7

        self.convolutions = []

        for i in range(len(features) - 1):
            relu = i < len(features) - 2
            conv = Block(features[i + 1], (features[i + 1] + features[i]) // 2, features[i], relu)
            setattr(self, 'conv{}'.format(i), conv)
            self.convolutions.append(conv)

        self.bn_in = nn.BatchNorm3d(1, affine=False)
        self.bn_out = nn.BatchNorm3d(number_of_classes, affine=True)

    def forward(self, x): # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.bn_in(x.contiguous())
        for conv in self.convolutions:
            x = conv(x)

        # [batch, features]
        x = x.mean(-1).squeeze(-1).mean(-1).squeeze(-1).mean(-1).squeeze(-1)
        x = self.bn_out(x.contiguous())
        return x


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = None

    def initialize(self, number_of_classes):
        self.cnn = CNN(number_of_classes)

    def get_cnn(self):
        if self.cnn is None:
            raise ValueError("Need to call initialize first")
        return self.cnn

    def get_batch_size(self):
        return 16

    def get_learning_rate(self, epoch):
        if epoch < 20:
            return 1e-1
        return 1e-2

    def load_files(self, files):
        images = np.array([np.load(file) for file in files], dtype=np.float32)
        images = images.reshape((-1, 1, 64, 64, 64))
        images = torch.FloatTensor(images)
        return images
