#pylint: disable=C,R,E1101
import torch
from se3_cnn.utils import time_logging

class TensorProduct:
    def __init__(self, enabled):
        '''
        :param enabled: list of triplets (multiplicity, dimension, bool)
        '''
        # super(TensorProduct, self).__init__()
        self.enabled = enabled

    def __call__(self, x):
        '''
        :param x: [batch, features, x, y, z]
        '''
        time = time_logging.start()
        nbatch = x.size(0)
        nx = x.size(2)
        ny = x.size(3)
        nz = x.size(4)
        ys = []

        begin = 0
        for m, dim, on in self.enabled:
            if on:
                for i in range(m):
                    s = slice(begin + i * dim, begin + (i + 1) * dim)
                    field = x[:, s] # [batch, features, x, y, z]
                    field = field.transpose(1, 4) # [batch, z, x, y, features]
                    field = field.contiguous()
                    a = field.view(-1, dim, 1) # [batch * z * x * y, features, 1]
                    b = field.view(-1, 1, dim) # [batch * z * x * y, 1, features]
                    field = a * b # [batch * z * x * y, features, features]
                    field = field.contiguous()
                    field = field.view(nbatch, nz, nx, ny, dim ** 2) # [batch, z, x, y, features]
                    field = field.transpose(1, 4) # [batch, features, x, y, z]

                    ys.append(field)

            begin += m * dim

        y = torch.cat(ys, dim=1)
        time_logging.end("tensor product", time)
        return y
