#pylint: disable=C,R,E1101
'''
Based on m3

+ scalar repr
+ batch norm
'''
import torch
import torch.nn as nn
from se3_cnn.convolution import SE3Convolution
from se3_cnn import SO3
from se3_cnn.train.model import Model
import logging
import numpy as np

logger = logging.getLogger("trainer")

class CNN(nn.Module):
    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        logger.info("Create CNN for classify %d classes", number_of_classes)

        representations = [
            [(1, SO3.repr1)], # 64
            [(6, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5), (1, SO3.repr7)], # 61
            [(6, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5), (1, SO3.repr7)], # 58
            [(6, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5), (1, SO3.repr7)], # 55
            [(6, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5), (1, SO3.repr7)], # 52
            [(number_of_classes, SO3.repr1)]] # 49

        self.convolutions = []

        for i in range(len(representations) - 1):
            non_lin = i < len(representations) - 2
            conv = SE3Convolution(4, representations[i + 1], representations[i], bias_relu=non_lin, norm_relu=non_lin, scalar_batch_norm=True)
            setattr(self, 'conv{}'.format(i), conv)
            self.convolutions.append(conv)

        self.bn_in = nn.BatchNorm3d(1, affine=True)
        self.bn_out = nn.BatchNorm3d(number_of_classes, affine=True)

        # self.bias = torch.nn.Parameter(torch.FloatTensor(number_of_classes))
        # self.bias.data[:] = 0

    def forward(self, x):
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.bn_in(x.contiguous())
        # logger.info("Mean = %f Std = %f", x.data.mean(), x.data.std())
        for conv in self.convolutions:
            # if x.size(1) == 35:
            #     logger.info("R1 Mean = %f Std = %f", x.data[:, :6].mean(), x.data[:, :6].std())
            #     logger.info("R3 Mean = %f Std = %f", x.data[:, 6:18].mean(), x.data[:, 6:18].std())
            #     logger.info("R5 Mean = %f Std = %f", x.data[:, 18:28].mean(), x.data[:, 18:28].std())
            #     logger.info("R7 Mean = %f Std = %f", x.data[:, 28:].mean(), x.data[:, 28:].std())
            x = conv(x)

        # logger.info("Mean = %f Std = %f", x.data.mean(), x.data.std())
        x = x.mean(-1).squeeze(-1).mean(-1).squeeze(-1).mean(-1).squeeze(-1) # [batch, features]
        x = self.bn_out(x.contiguous())
        # logger.info("Mean = %f Std = %f", x.data.mean(), x.data.std())
        # x = x + self.bias.view(1, -1).expand_as(x)
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
        return 1e-3

    def load_files(self, files):
        images = np.array([np.load(file) for file in files], dtype=np.float32)
        images = images.reshape((-1, 1, 64, 64, 64))
        images = torch.FloatTensor(images)
        return images
