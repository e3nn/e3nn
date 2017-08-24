#pylint: disable=C,R,E1101
'''
Based on m7

+ not equivariant
'''
import torch
import torch.nn as nn
import math
from util_cnn.model import Model
import logging
import numpy as np

logger = logging.getLogger("trainer")

class CNN(nn.Module):
    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        self.features = [
            1,
            6 + 4 * 3 + 2 * 5 + 1 * 7,
            6 + 4 * 3 + 2 * 5 + 1 * 7,
            6 + 4 * 3 + 2 * 5 + 1 * 7,
            6 + 4 * 3 + 2 * 5 + 1 * 7,
            number_of_classes]

        for i in range(len(self.features) - 1):
            weights = torch.nn.Parameter(torch.FloatTensor(self.features[i+1], self.features[i], 4, 4, 4))
            weights.data.normal_(0, 1 / math.sqrt(self.features[i] * 4 * 4 * 4))
            setattr(self, 'weights{}'.format(i), weights)

            bias = torch.nn.Parameter(torch.FloatTensor(self.features[i+1]))
            bias.data[:] = 0
            setattr(self, 'bias{}'.format(i), bias)

    def forward(self, x):
        '''
        :param x: [batch, features, x, y, z]
        '''
        for i in range(len(self.features) - 1):
            x = torch.nn.functional.conv3d(x, getattr(self, 'weights{}'.format(i)))
            x = x + getattr(self, 'bias{}'.format(i)).view(1, -1, 1, 1, 1).expand_as(x)
            if i < len(self.features) - 2:
                x = torch.nn.functional.relu(x)

        x = x.mean(-1).mean(-1).mean(-1) # [batch, features]
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

    def get_batch_size(self, epoch=None):
        return 16

    def get_learning_rate(self, epoch):
        return 1e-3

    def load_files(self, files):
        images = np.array([np.load(file)['arr_0'] for file in files], dtype=np.float32)
        images = images.reshape((-1, 1, 64, 64, 64))
        images = torch.FloatTensor(images)
        return images
