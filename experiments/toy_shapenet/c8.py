# pylint: disable=C,R,E1101
'''
Based on c7

+ less stride
+ more layers
'''
import torch
import torch.nn as nn
from se3_cnn.convolution import SE3Convolution
from se3_cnn.non_linearities.tensor_product import TensorProduct
from se3_cnn import SO3
from util_cnn.model import Model
from util_cnn import time_logging
import logging
import numpy as np

logger = logging.getLogger("trainer")


class Block(nn.Module):
    def __init__(self, scalar_out, vector, scalar_in, relu):
        super().__init__()
        self.conv1 = SE3Convolution(7, 3, [(scalar_out, SO3.repr1), (vector, SO3.repr3)], [(scalar_in, SO3.repr1)],
            bias_relu=True,
            norm_relu=False,
            scalar_batch_norm=False,
            stride=2,
            padding=5)
        self.tensor = TensorProduct([(scalar_out, 1, False), (vector, 3, True)])
        self.conv2 = SE3Convolution(7, 3, [(scalar_out, SO3.repr1)], [(scalar_out, SO3.repr1), (vector, SO3.repr3), (vector, SO3.repr3x3)],
            bias_relu=relu,
            norm_relu=False,
            scalar_batch_norm=True,
            stride=1,
            padding=3)

    def forward(self, s): # pylint: disable=W
        sv = self.conv1(s) # bias and relu for scalar part
        t = self.tensor(sv)
        svt = torch.cat([sv, t], dim=1)
        s = self.conv2(svt) # BN, bias and relu
        return s

class CNN(nn.Module):

    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        logger.info("Create CNN for classify %d classes", number_of_classes)

        features = [(1, 2), # 64
            (4, 4), # (64 + 2*5 - 6) / 2 + 2*3 - 6 = (64+4)/2 = 64/2 + 2 = 34
            (16, 4), # 34/2 + 2 = 19
            (32, 4), # 19/2 + 2 = 11
            (number_of_classes, )]  # 11/2 + 2 = 7

        self.convolutions = []

        for i in range(len(features) - 1):
            relu = i < len(features) - 2
            conv = Block(features[i + 1][0], features[i][1], features[i][0], relu)
            setattr(self, 'conv{}'.format(i), conv)
            self.convolutions.append(conv)

        self.bn_in = nn.BatchNorm3d(1, affine=False)
        self.bn_out = nn.BatchNorm1d(number_of_classes, affine=True)

    def forward(self, x): # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.bn_in(x.contiguous())
        t = time_logging.start()
        for i, conv in enumerate(self.convolutions):
            x = conv(x)
            t = time_logging.end("block {}".format(i), t)

        # [batch, features]
        x = x.mean(-1).mean(-1).mean(-1)
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
