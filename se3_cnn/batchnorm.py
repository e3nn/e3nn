# pylint: disable=C,R,E1101
import torch
import torch.nn as nn


class SE3BatchNorm(nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1, affine=True):
        '''
        :param Rs: list of tuple (multiplicity, dimension)
        '''
        super().__init__()

        self.Rs = [(m, d) for m, d in Rs if m * d > 0]
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        num_scalar = sum(m for m, d in Rs if d == 1)
        num_features = sum(m for m, d in Rs)

        self.register_buffer('running_mean', torch.zeros(num_scalar))
        self.register_buffer('running_var', torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def __repr__(self):
        return "{} (Rs={}, eps={}, momentum={})".format(
            self.__class__.__name__,
            self.Rs,
            self.eps,
            self.momentum)

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, stacked feature, x, y, z]
        '''

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0
        for m, d in self.Rs:
            field = input[:, ix: ix + m * d]  # [batch, feature * repr, x, y, z]
            ix += m * d
            field = field.contiguous().view(input.size(0), m, d, -1)  # [batch, feature, repr, x * y * z]

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(0).mean(-1).view(-1)  # [feature]
                    self.running_mean[irm: irm + m] = (1 - self.momentum) * self.running_mean[irm: irm + m] + self.momentum * field_mean.data
                else:
                    field_mean = torch.autograd.Variable(self.running_mean[irm: irm + m])
                irm += m
                field = field - field_mean.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                field_norm = field_norm.mean(0).mean(-1)  # [feature]
                self.running_var[irv: irv + m] = (1 - self.momentum) * self.running_var[irv: irv + m] + self.momentum * field_norm.data
            else:
                field_norm = torch.autograd.Variable(self.running_var[irv: irv + m])
            irv += m

            field_norm = (field_norm + self.eps).pow(-0.5).view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            if self.affine:
                weight = self.weight[iw: iw + m]  # [feature]
                iw += m
                field_norm = field_norm * weight.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            field = field * field_norm  # [batch, feature, repr, x * y * z]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + m]  # [feature]
                ib += m
                field = field + bias.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            fields.append(field.view(input.size(0), m * d, *input.size()[2:]))

        assert ix == input.size(1)
        if self.training:
            assert irm == self.running_mean.size(0)
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.size(0)

        return torch.cat(fields, dim=1)  # [batch, stacked feature, x, y, z]


def test_batchnorm():
    bn = SE3BatchNorm([(3, 1), (4, 3), (1, 5)])
    bn.bias.data[0] = 42
    bn.weight.data[1] = 32

    x = torch.autograd.Variable(torch.randn(16, 3 + 12 + 5, 10, 10, 10) * 3 + 12)

    bn.train()
    y = bn(x)

    bn.eval()
    z = bn(x)

    return y, z
