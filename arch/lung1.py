# pylint: disable=C,R,E1101
'''
Detect Lung nodule
'''
import torch
import torch.nn as nn
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.bn_conv import SE3BNConvolution
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn import SO3
from util_cnn.model_backup import ModelBackup

from util_cnn import time_logging
import logging
import numpy as np

logger = logging.getLogger("trainer")


class Block(nn.Module):
    def __init__(self, repr_in, repr_out, non_linearities, stride):
        super().__init__()
        self.repr_out = repr_out
        self.bn_conv = SE3BNConvolution(
            size=7,
            radial_amount=3,
            Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5)],
            Rs_out=[(repr_out[0], SO3.repr1), (repr_out[1], SO3.repr3), (repr_out[2], SO3.repr5)],
            stride=stride,
            padding=3,
            momentum=0.01,
            mode='maximum')

        self.non_linearities = non_linearities
        if non_linearities:
            self.relu = BiasRelu([(repr_out[0], True), (repr_out[1] * 3, False), (repr_out[2] * 5, False)], normalize=False)
            if repr_out[1] + repr_out[2] > 0:
                self.bn_conv_gate = SE3BNConvolution(
                    size=7,
                    radial_amount=3,
                    Rs_in=[(repr_in[0], SO3.repr1), (repr_in[1], SO3.repr3), (repr_in[2], SO3.repr5)],
                    Rs_out=[(repr_out[1] + repr_out[2], SO3.repr1)],
                    stride=stride,
                    padding=3,
                    momentum=0.01,
                    mode='maximum')
                self.relu_gate = BiasRelu([(repr_out[1] + repr_out[2], True)], normalize=False)

    def forward(self, x): # pylint: disable=W
        y = self.bn_conv(x)

        if self.non_linearities:
            y = self.relu(y)

            u = self.bn_conv_gate(x)
            u = self.relu_gate(u)

            nbatch = y.size(0)
            nx = y.size(2)
            ny = y.size(3)
            nz = y.size(4)

            zs = [y[:, :self.repr_out[0]]]

            if self.repr_out[1] + self.repr_out[2] > 0:
                begin_y = self.repr_out[0]
                begin_u = 0

                for (m, dim) in [(self.repr_out[1], 3), (self.repr_out[2], 5)]:
                    if m == 0:
                        continue
                    field_y = y[:, begin_y: begin_y + m * dim] # [batch, feature * repr, x, y, z]
                    field_y = field_y.contiguous()
                    field_y = field_y.view(nbatch, m, dim, nx, ny, nz) # [batch, feature, repr, x, y, z]
                    field_u = u[:, begin_u: begin_u + m] # [batch, feature, x, y, z]
                    field_u = field_u.contiguous()
                    field_u = field_u.view(nbatch, m, 1, nx, ny, nz) # [batch, feature, repr, x, y, z]
                    field = field_y * field_u # [batch, feature, repr, x, y, z]
                    field = field.view(nbatch, m * dim, nx, ny, nz) # [batch, feature * repr, x, y, z]
                    zs.append(field)

                    begin_y += m * dim
                    begin_u += m

            z = torch.cat(zs, dim=1)
            return z
        else:
            return y

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
            (16, 8, 1), # 8
            (2, 0, 0) # 4
        ]
        self.block_params = [
            {'stride': 2, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 1, 'non_linearities': True},
            {'stride': 2, 'non_linearities': True},
            {'stride': 2, 'non_linearities': True},
            {'stride': 2, 'non_linearities': False},
        ]

        assert len(self.block_params) + 1 == len(self.features)

        for i in range(len(self.block_params)):
            block = Block(self.features[i], self.features[i + 1], **self.block_params[i])
            setattr(self, 'block{}'.format(i), block)

    def forward(self, inp): # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = inp

        t = time_logging.start()
        for i in range(len(self.block_params)):
            logger.info("%d: %f +- %f", i, x.data.mean(), x.data.std())

            block = getattr(self, 'block{}'.format(i))
            x = block(x)
            t = time_logging.end("block {}".format(i), t)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x*y*z]
        x = x.sum(-1) # [batch, features]

        return x


class MyModel(ModelBackup):
    def __init__(self):
        super().__init__(
            success_factor=1,
            decay_factor=2 ** (-1/4 * 1/6),
            reject_factor=2 ** (-1),
            reject_ratio=2,
            min_learning_rate=1e-4,
            max_learning_rate=0.2,
            initial_learning_rate=2e-3)

    def initialize(self, **kargs):
        self.cnn = CNN()
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def get_batch_size(self, epoch=None):
        return 16

    def create_train_batches(self, epoch, files, labels):
        labels = np.array(labels)
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]

        batches = []
        for i in pos:
            batches.append([i] + list(np.random.choice(neg, size=15, replace=False)))
        return batches

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def get_learning_rate(self, epoch):
        for module in self.cnn.modules():
            if isinstance(module, SE3BatchNorm):
                module.momentum = 0.01 * (0.1 ** epoch)

        return super().get_learning_rate(epoch)

    def load_files(self, files):
        import glob
        n = 64

        inputs = np.zeros((len(files), 1, n, n, n), dtype=np.float32)

        for i, f in enumerate(files):
            f = np.load(f)
            name = str(f['name'])
            x, y, z = f['position']
            raw_file = next(f for f in glob.glob('subset*/*.mhd') if name in f)
            inputs[i] = self.extract(raw_file, x, y, z)

        inputs = np.clip(inputs, -1000, 500)
        inputs = (inputs + 1000) / 1500

        return inputs

    def extract(self, file, x, y, z):
        import SimpleITK as sitk
        from scipy import ndimage

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

        return output
