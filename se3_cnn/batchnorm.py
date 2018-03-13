# pylint: disable=C,R,E1101
import torch


class SE3BatchNorm(torch.nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1, mode='normal'):
        '''
        :param Rs: list of tuple (multiplicity, dimension)
        '''
        super().__init__()

        self.Rs = [(m, d) for m, d in Rs if m * d > 0]
        self.num_features = sum([m for m, d in Rs])

        self.eps = eps
        self.momentum = momentum
        self.mode = mode
        self.register_buffer('running_var', torch.ones(self.num_features))
        self.weight = torch.nn.Parameter(torch.ones(self.num_features))
        self.reset_parameters()

    def __repr__(self):
        return "{} (Rs={}, eps={}, momentum={}, mode={})".format(
            self.__class__.__name__,
            self.Rs,
            self.eps,
            self.momentum,
            self.mode)

    def reset_parameters(self):
        self.running_var.fill_(1)

    def update_statistics(self, x, divisor=None):
        '''
        update self.running_var using x

        :param x: Tensor [batch, feature, x, y, z]
        :param divisor: Tensor same size as self.running_var

        divisor is needed by SE3ConvolutionBN
        '''
        if self.training and self.momentum > 0:
            begin1 = 0
            begin2 = 0
            for m, d in self.Rs:
                y = x[:, begin1: begin1 + m * d]  # [batch, feature * repr, x, y, z]
                begin1 += m * d
                y = y.contiguous().view(x.size(0), m, d, -1)  # [batch, feature, repr, x * y * z]

                y = torch.sum(y ** 2, dim=2)  # [batch, feature, x * y * z]

                if divisor is not None:
                    y = y / (divisor[begin2: begin2 + m] ** 2 + self.eps).view(1, -1, 1)  # [batch, feature, x * y * z]

                if self.mode == 'normal':
                    y = y.mean(-1).mean(0)  # [feature]
                elif self.mode == 'ignore_zeros':
                    mask = torch.abs(y) > self.eps  # [batch, feature, x * y * z]
                    number = mask.sum(-1).sum(0)  # [feature]
                    y = y.sum(-1).sum(0)  # [feature]
                    y = y / (number.float() + self.eps)
                elif self.mode == 'maximum':
                    y = y.max(-1)[0].mean(0)  # [feature]
                else:
                    raise ValueError("no mode named \"{}\"".format(self.mode))

                self.running_var[begin2: begin2 + m] = (1 - self.momentum) * self.running_var[begin2: begin2 + m] + self.momentum * y
                begin2 += m

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, feature, x, y, z]
        '''
        self.update_statistics(x.data)

        ys = []
        begin1 = 0
        begin2 = 0
        for m, d in self.Rs:
            y = x[:, begin1: begin1 + m * d]
            begin1 += m * d
            y = y.contiguous().view(x.size(0), m, d, *x.size()[2:])  # [batch, feature, repr, x, y, z]

            factor = 1 / (self.running_var[begin2: begin2 + m] + self.eps) ** 0.5
            weight = self.weight[begin2: begin2 + m]

            begin2 += m

            y = y * (torch.autograd.Variable(factor) * weight).view(1, -1, 1, 1, 1, 1)
            ys.append(y.view(x.size(0), m * d, *x.size()[2:]))

        y = torch.cat(ys, dim=1)
        return y


def test_batchnorm():
    bn = SE3BatchNorm([(3, 1), (4, 3), (1, 5)])

    x = torch.autograd.Variable(torch.randn(16, 3 + 12 + 5, 10, 10, 10))

    y = bn(x)
    return y
