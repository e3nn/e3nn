#pylint: disable=C,R,E1101
import torch
import torch.nn as nn
from se3_cnn.convolution import SE3Convolution
from se3_cnn import SO3
from se3_cnn.train.model import Model
import logging
import numpy as np

logger = logging.getLogger("trainer")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        representations = [
            [(1, SO3.repr1)],
            [(8, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5)],
            [(8, SO3.repr1), (4, SO3.repr3), (2, SO3.repr5)],
            [(2, SO3.repr1)]]

        self.convolutions = []

        for i in range(len(representations) - 1):
            conv = SE3Convolution(4, representations[i + 1], representations[i], bias_relu=i < len(representations) - 2)
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
        self.cnn = CNN()

    def get_cnn(self):
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
