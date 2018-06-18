# pylint: disable=C,R,E1101
'''
Best SE3 equivariant architecture found for shapenet
classification of 6 classes

c21
'''
import torch
from se3cnn.blocks.tensor_product import TensorProductBlock
from util_cnn.model_backup import ModelBackup

from util_cnn import time_logging
import logging
import numpy as np

logger = logging.getLogger("trainer")


class CNN(torch.nn.Module):

    def __init__(self, number_of_classes):
        super(CNN, self).__init__()

        logger.info("Create CNN for classify %d classes", number_of_classes)

        features = [(1, 0, 0), # 64
            (4, 2, 0), # 32
            (10, 5, 1), # 32
            (10, 5, 1), # 16
            (12, 6, 1), # 16
            (12, 6, 2), # 8
            (14, 7, 2), # 8
            (number_of_classes, 0, 0)]  # 4

        self.convolutions = []

        for i in range(len(features) - 1):
            relu = i < len(features) - 2
            conv = TensorProductBlock(features[i], features[i + 1], relu, 2 if i % 2 == 0 else 1)
            setattr(self, 'conv{}'.format(i), conv)
            self.convolutions.append(conv)

        self.bn_in = torch.nn.BatchNorm3d(1, affine=False)
        self.bn_out = torch.nn.BatchNorm1d(number_of_classes, affine=True)

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
        x = x.view(x.size(0), x.size(1), -1).max(-1)[0]
        x = self.bn_out(x.contiguous())
        return x


class MyModel(ModelBackup):
    def __init__(self):
        super().__init__(
            success_factor=2 ** (1/6),
            decay_factor=2 ** (-1/4 * 1/6),
            reject_factor=2 ** (-1),
            reject_ratio=1.5,
            min_learning_rate=1e-4,
            max_learning_rate=0.2,
            initial_learning_rate=1e-2)

    def initialize(self, number_of_classes): # pylint: disable=W0221
        self.cnn = CNN(number_of_classes)
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def get_batch_size(self, epoch=None):
        return 16

    def load_files(self, files):
        images = np.array([np.load(file)['arr_0'] for file in files], dtype=np.float32)
        images = images.reshape((-1, 1, 64, 64, 64))
        images = (images - 0.02267) / 0.14885
        images = torch.FloatTensor(images)
        return images
