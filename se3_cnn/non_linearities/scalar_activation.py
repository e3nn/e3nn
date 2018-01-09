# pylint: disable=C,R,E1101
import torch


class ScalarActivation(torch.nn.Module):
    def __init__(self, enable, activation, bias=True):
        '''
        Can be used only with scalar fields

        :param enable: list of tuple (dimension, boolean on/off)
        :param activation: function that takes in input a torch.autograd.Variable
        :param bool bias: add a bias before the applying the activation
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
        if bias and nbias > 0:
            self.bias = torch.nn.Parameter(torch.FloatTensor(nbias))
            self.bias.data[:] = 0
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        xs = []
        begin1 = 0
        begin2 = 0

        for d, on in self.enable:
            x = input[:, begin1:begin1 + d]

            if on:
                if self.bias is not None:
                    x = x + self.bias[begin2:begin2 + d].view(1, -1, 1, 1, 1)
                    begin2 += d

                x = self.activation(x)

            xs.append(x)

            begin1 += d

        assert begin1 == input.size(1)
        assert self.bias is None or begin2 == self.bias.size(0)

        return torch.cat(xs, dim=1)
