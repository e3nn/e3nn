#pylint: disable=C,R,E1101
'''
Best architecture found for shapenet
classification of 6 classes
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
            1, # 64
            6 + 4 * 3 + 2 * 5 + 1 * 7, # =35 # 33
            6 + 4 * 3 + 2 * 5 + 1 * 7, # 17
            6 + 4 * 3 + 2 * 5 + 1 * 7, # 9
            6 + 4 * 3 + 2 * 5 + 1 * 7, # 5
            number_of_classes]

        for i in range(len(self.features) - 1):
            weights = torch.nn.Parameter(torch.FloatTensor(self.features[i+1], self.features[i], 4, 4, 4))
            weights.data.normal_(0, 1 / math.sqrt(self.features[i] * 4 * 4 * 4))
            setattr(self, 'weights{}'.format(i), weights)

            bn = nn.BatchNorm3d(self.features[i+1], affine=True)
            setattr(self, 'bn_bias{}'.format(i), bn)

        self.bn_in = nn.BatchNorm3d(1, affine=True)
        self.bn_out = nn.BatchNorm1d(number_of_classes, affine=True)

    def forward(self, x): # pylint: disable=W0221
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.bn_in(x.contiguous())
        for i in range(len(self.features) - 1):
            x = torch.nn.functional.conv3d(x, getattr(self, 'weights{}'.format(i)), stride=2, padding=2)
            x = getattr(self, 'bn_bias{}'.format(i))(x)
            if i < len(self.features) - 2:
                x = torch.nn.functional.relu(x)

        x = x.mean(-1).mean(-1).mean(-1) # [batch, features]
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

    def get_batch_size(self, epoch=None):
        return 16

    def get_learning_rate(self, epoch):
        return 1e-3

    def load_files(self, files):
        images = np.array([np.load(file)['arr_0'] for file in files], dtype=np.float32)
        images = images.reshape((-1, 1, 64, 64, 64))
        images = torch.FloatTensor(images)
        return images
