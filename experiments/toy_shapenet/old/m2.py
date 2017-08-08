#pylint: disable=C,R,E1101
'''
Based on m1

+ tensor products
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        representations = [
            [(1, SO3.repr1)], # -> Conv =>
            [(1, SO3.repr3), (1, SO3.repr5)], # -> TensorProduct =>
            [(1, SO3.repr3), (1, SO3.repr5), (1, SO3.repr3x3), (1, SO3.repr5x5)], # -> Conv =>
            [(1, SO3.repr3), (1, SO3.repr5)], # -> TensorProduct =>
            [(1, SO3.repr3), (1, SO3.repr5), (1, SO3.repr3x3), (1, SO3.repr5x5)], # -> Conv =>
            [(2, SO3.repr1)]]

        size = 4
        self.conv1 = SE3Convolution(size, representations[1], representations[0], bias_relu=True)
        self.tensor_product1 = TensorProduct([(m, SO3.dim(d), True) for m, d in representations[1]])

        self.conv2 = SE3Convolution(size, representations[3], representations[2], bias_relu=True)
        self.tensor_product2 = TensorProduct([(m, SO3.dim(d), True) for m, d in representations[3]])

        self.conv3 = SE3Convolution(size, representations[5], representations[4], bias_relu=True)


    def forward(self, x):
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.conv1(x)
        xx = self.tensor_product1(x)
        x = torch.cat([x, xx], dim=1)

        x = self.conv2(x)
        xx = self.tensor_product2(x)
        x = torch.cat([x, xx], dim=1)

        x = self.conv3(x)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x * y * z]
        x = x.mean(-1).squeeze(-1) # [batch, features]
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
        images = torch.FloatTensor(images)
        return images
