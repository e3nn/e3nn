#pylint: disable=C,R,E1101
import torch
from torch.nn.parameter import Parameter

class BiasRelu(torch.nn.Module):
    def __init__(self, enable, batch_norm=False):
        '''
        :param enable: list of tuple (dimension, boolean)

        If boolean is True a bias and relu will be applied
        '''
        super(BiasRelu, self).__init__()

        self.enable = []
        for d, on in enable:
            if d == 0:
                continue

            if len(self.enable) > 0 and self.enable[-1][1] == on:
                self.enable[-1] = (self.enable[-1][0] + d, on)
            else:
                self.enable.append((d, on))

        self.batch_norms = None
        if batch_norm:
            self.batch_norms = []
            for d, on in self.enable:
                if on:
                    bn = torch.nn.BatchNorm3d(d, affine=False)
                    setattr(self, "bn{}".format(len(self.batch_norms)), bn)
                    self.batch_norms.append(bn)

        nbias = sum([d for d, on in self.enable if on])
        self.bias = Parameter(torch.FloatTensor(nbias)) if nbias > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data[:] = 0

    def forward(self, input): # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        if self.bias is None:
            return input

        xs = []
        begin1 = 0
        begin2 = 0
        bn_counter = 0

        for d, on in self.enable:
            x = input[:, begin1:begin1 + d]

            if on:
                if self.batch_norms is not None:
                    x = self.batch_norms[bn_counter](x.contiguous())
                    bn_counter += 1
                x = x + self.bias[begin2:begin2 + d].view(1, -1, 1, 1, 1).expand_as(x)
                x = torch.nn.functional.relu(x)
                x.sub_(0.3989422804014327)
                x.mul_(1.712858550449663)
                begin2 += d

            xs.append(x)

            begin1 += d

        assert begin1 == input.size(1)
        assert begin2 == self.bias.size(0)

        return torch.cat(xs, dim=1)
