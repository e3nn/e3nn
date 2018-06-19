# pylint: disable=C,R,E1101
'''
Detect Lung nodule
'''
import torch
import torch.nn as nn
from se3cnn.batchnorm import SE3BatchNorm
from se3cnn.blocks.highway import HighwayBlock
from util_cnn.model_backup import ModelBackup
import glob
import os

from util_cnn import time_logging
import logging
import numpy as np

logger = logging.getLogger("trainer")


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.features = [
            (1, 0, 0), # 64
            (10, 3, 0), # 32
            (10, 3, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 32
            (16, 8, 1), # 16
            (1, 0, 0) # 8
        ]
        self.block_params = [
            {'stride': 2, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 2, 'non_linearities': True},
            {'stride': 2, 'non_linearities': False},
        ]

        assert len(self.block_params) + 1 == len(self.features)

        for i in range(len(self.block_params)):
            block = HighwayBlock(self.features[i], self.features[i + 1], **self.block_params[i])
            setattr(self, 'block{}'.format(i), block)

    def forward(self, inp): # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = inp

        t = time_logging.start()
        for i in range(len(self.block_params)):
            #logger.info("%d: %f +- %f", i, x.data.mean(), x.data.std())

            block = getattr(self, 'block{}'.format(i))
            x = block(x)
            t = time_logging.end("block {}".format(i), t)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x*y*z]
        x = x.sum(-1) # [batch, features]

        return x


mhd_files = glob.glob('subset*/*.mhd')

class MyModel(ModelBackup):
    def __init__(self):
        super().__init__(
            success_factor=1,
            decay_factor=2 ** (-1/4 * 1/6),
            reject_factor=2 ** (-1),
            reject_ratio=2,
            min_learning_rate=1e-5,
            max_learning_rate=1e-2,
            initial_learning_rate=1e-3)

    def initialize(self, **kargs):
        self.cnn = CNN()
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def get_batch_size(self, epoch=None):
        return 16

    def create_train_batches(self, epoch, files, labels):
        labels = np.array(labels)
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]

        np.random.shuffle(pos)
        np.random.shuffle(neg)

        batches = []
        for i in range(len(pos) // 8):
            batch = list(pos[8*i: 8*i+8]) + list(neg[8*i: 8*i+8])
            batches.append(batch)
        return batches

    def get_criterion(self):
        class Crit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._crit = torch.nn.BCEWithLogitsLoss()
            def forward(self, logit, target): # pylint: disable=W0221
                target = target.float().view(target.size(0), 1)
                return self._crit(logit, target)

        return Crit()

    def get_learning_rate(self, epoch):
        for module in self.cnn.modules():
            if isinstance(module, SE3BatchNorm):
                module.momentum = 0.01 * (0.1 ** epoch)

        return super().get_learning_rate(epoch)

    def load_files(self, files):
        n = 64

        inputs = np.zeros((len(files), 1, n, n, n), dtype=np.float32)

        for i, f in enumerate(files):
            f = np.load(f)
            name = str(f['name'])
            x, y, z = f['position']
            raw_file = next(f for f in mhd_files if name in f)
            inputs[i] = self.extract(raw_file, x, y, z)

        inputs = np.clip(inputs, -1000, 500)
        inputs = (inputs + 1000) / 1500

        return inputs

    def extract(self, file, x, y, z):
        import SimpleITK as sitk
        from scipy import ndimage

        cache_file = 'cache/' + file.split('/')[-1][:-4] + '_x{}_y{}_z{}'.format(int(round(x * 1000)), int(round(y * 1000)), int(round(z * 1000))).replace('-', 'm') + '.npy'

        if os.path.isfile(cache_file):
            return np.load(cache_file)

        img = sitk.ReadImage(file)
        nda = sitk.GetArrayFromImage(img)

        spacing = np.array(img.GetSpacing())

        size = 64
        precision = 0.5

        # select the candidate region
        original_half_size = np.ceil(0.5 * size * precision / spacing).astype(np.int32)
        original_center = np.array(img.TransformPhysicalPointToIndex((x, y, z)))
        original_size = np.array(img.GetSize())

        a, b = original_center - original_half_size, original_center + original_half_size
        offset_due_to_border = np.maximum(-a, 0) + np.minimum(original_size - b, 0)
        a += offset_due_to_border
        b += offset_due_to_border
        sub_nda = nda[a[2]:b[2], a[1]:b[1], a[0]:b[0]]

        # rescale voxel size to `precision`
        zoom = spacing / precision
        output = ndimage.zoom(sub_nda, zoom=(zoom[2], zoom[1], zoom[0]))

        # crop excess
        shape = np.array(output.shape)
        a = (shape - size) // 2
        output = output[a[0]:a[0] + size, a[1]:a[1] + size, a[2]:a[2] + size]

        assert output.shape[0] == size
        assert output.shape[1] == size
        assert output.shape[2] == size

        np.save(cache_file, output)

        return output
