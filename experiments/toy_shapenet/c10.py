# pylint: disable=C,R,E1101
'''
Based on c9

+ more layers
'''
import torch
import torch.nn as nn
from se3_cnn.convolution import SE3Convolution
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn.non_linearities.tensor_product import TensorProduct
from se3_cnn import SO3
from se3_cnn.train.model import Model
from se3_cnn.utils import time_logging
import logging
import numpy as np

logger = logging.getLogger("trainer")


class Block(nn.Module):
    def __init__(self, scalar_out, vector_out, scalar_in, vector_in, relu, stride):
        super().__init__()
        self.tensor = TensorProduct([(scalar_in, 1, False), (vector_in, 3, True)]) if vector_in > 0 else None
        self.conv = SE3Convolution(7, 3,
            [(scalar_out, SO3.repr1), (vector_out, SO3.repr3)],
            [(scalar_in, SO3.repr1), (vector_in, SO3.repr3), (vector_in, SO3.repr3x3)],
            stride=stride,
            padding=3)
        self.bn = SE3BatchNorm([(scalar_out, 1), (vector_out, 3)])
        self.relu = BiasRelu([(scalar_out, True), (vector_out * 3, False)]) if relu else None

    def forward(self, sv): # pylint: disable=W
        if self.tensor is not None:
            t = self.tensor(sv)
            svt = torch.cat([sv, t], dim=1)
        else:
            svt = sv
        sv = self.conv(svt) # only convolution
        sv = self.bn(sv)
        if self.relu is not None:
            sv = self.relu(sv)
        return sv

class CNN(nn.Module):

    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        logger.info("Create CNN for classify %d classes", number_of_classes)

        features = [(1, 0), # 64
            (4, 2), # 32
            (10, 5), # 32
            (10, 5), # 16
            (12, 6), # 16
            (12, 6), # 8
            (14, 7), # 8
            (number_of_classes, 0)]  # 4

        self.convolutions = []

        for i in range(len(features) - 1):
            relu = i < len(features) - 2
            conv = Block(features[i + 1][0], features[i + 1][1], features[i][0], features[i][1], relu, 2 if i % 2 == 0 else 1)
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
