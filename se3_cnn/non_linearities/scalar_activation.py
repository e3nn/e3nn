#pylint: disable=C,R,E1101
import torch
from torch.nn.parameter import Parameter

class BiasRelu(torch.nn.Module):
    def __init__(self, enable, normalize=True):
        '''
        :param enable: list of tuple (dimension, boolean)

        If boolean is True a bias and relu will be applied
        '''
        super().__init__()

        self.enable = []
        for d, on in enable:
            if d == 0:
                continue

            if len(self.enable) > 0 and self.enable[-1][1] == on:
                self.enable[-1] = (self.enable[-1][0] + d, on)
            else:
                self.enable.append((d, on))

        nbias = sum([d for d, on in self.enable if on])
        if nbias > 0:
            self.bias = Parameter(torch.FloatTensor(nbias))
            self.bias.data[:] = 0
        else:
            self.bias = None

        self.normalize = normalize

    def forward(self, input): # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        if self.bias is None:
            return input

        xs = []
        begin1 = 0
        begin2 = 0

        for d, on in self.enable:
            x = input[:, begin1:begin1 + d]

            if on:
                x = x + self.bias[begin2:begin2 + d].view(1, -1, 1, 1, 1).expand_as(x)
                x = torch.nn.functional.relu(x)
                if self.normalize:
                    x.sub_(0.3989422804014327)
                    x.mul_(1.712858550449663)
                begin2 += d

            xs.append(x)

            begin1 += d

        assert begin1 == input.size(1)
        assert begin2 == self.bias.size(0)

        return torch.cat(xs, dim=1)
