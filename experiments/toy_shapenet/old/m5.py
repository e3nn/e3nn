#pylint: disable=C,R,E1101
'''
Based on m1

+ only scalar
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
            [(1, SO3.repr1)],
            [(16, SO3.repr1)],
            [(16, SO3.repr1)],
            [(number_of_classes, SO3.repr1)]]

        self.convolutions = []

        for i in range(len(representations) - 1):
            non_lin = i < len(representations) - 2
            conv = SE3Convolution(4, representations[i + 1], representations[i], bias_relu=non_lin)
            setattr(self, 'conv{}'.format(i), conv)
            self.convolutions.append(conv)

    def forward(self, x):
        '''
        :param x: [batch, features, x, y, z]
        '''
        for conv in self.convolutions:
            x = conv(x)

        x = x.mean(-1).squeeze(-1).mean(-1).squeeze(-1).mean(-1).squeeze(-1) # [batch, features]
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
        images = torch.autograd.Variable(torch.FloatTensor(images))
        return images
