# pylint: disable=C,R,E1101
import torch


class ScalarActivation(torch.nn.Module):
    def __init__(self, enable, bias=True, inplace=False):
        '''
        Can be used only with scalar fields

        :param enable: list of tuple (dimension, activation function or None)
        :param bool bias: add a bias before the applying the activation
        '''
        super().__init__()

        self.inplace = inplace
        self.enable = []
        for d, act in enable:
            if d == 0:
                continue

            if self.enable and self.enable[-1][1] is act:
                self.enable[-1] = (self.enable[-1][0] + d, act)
            else:
                self.enable.append((d, act))

        nbias = sum([d for d, act in self.enable if act is not None])
        if bias and nbias > 0:
            self.bias = torch.nn.Parameter(torch.zeros(nbias))
        else:
            self.bias = None

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        begin1 = 0
        begin2 = 0

        if self.inplace:
            output = input
        else:
            output = torch.empty_like(input)

        for d, act in self.enable:
            x = input[:, begin1:begin1 + d]

            if act is not None:
                if self.bias is not None:
                    x = x + self.bias[begin2:begin2 + d].view(1, -1, 1, 1, 1)
                    begin2 += d

                x = act(x)

            if not self.inplace or act is not None:
                output[:, begin1:begin1 + d] = x

            begin1 += d

        assert begin1 == input.size(1)
        assert self.bias is None or begin2 == self.bias.size(0)

        return output
