#pylint: disable=C,R,E1101
import torch
from se3_cnn.utils import time_logging

class SE3BatchNorm(torch.nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1):
        '''
        :param Rs: list of tuple (multiplicity, dimension)
        '''
        super().__init__()

        self.Rs = [(m, d) for m, d in Rs if m * d > 0]
        self.num_features = sum([m for m, d in Rs])

        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_var', torch.ones(self.num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def forward(self, x): # pylint: disable=W
        '''
        :param x: [batch, feature, x, y, z]
        '''
        time = time_logging.start()
        if self.training:
            begin1 = 0
            begin2 = 0
            for m, d in self.Rs:
                y = x.data[:, begin1: begin1 + m * d] # [batch, feature * repr, x, y, z]
                begin1 += m * d
                y = y.contiguous().view(x.size(0), m, d, *x.size()[2:]) # [batch, feature, repr, x, y, z]

                y = torch.sum(y * y, dim=2) # [batch, feature, x, y, z]
                y = y.mean(-1).mean(-1).mean(-1).mean(0) # [feature]

                self.running_var[begin2: begin2 + m] = (1 - self.momentum) * self.running_var[begin2: begin2 + m] + self.momentum * y
                begin2 += m

        ys = []
        begin1 = 0
        begin2 = 0
        for m, d in self.Rs:
            y = x[:, begin1: begin1 + m * d]
            begin1 += m * d
            y = y.contiguous().view(x.size(0), m, d, *x.size()[2:]) # [batch, feature, repr, x, y, z]

            factor = 1 / (self.running_var[begin2: begin2 + m] + self.eps) ** 0.5
            begin2 += m

            y = y * torch.autograd.Variable(factor).view(1, -1, 1, 1, 1, 1)
            ys.append(y.view(x.size(0), m * d, *x.size()[2:]))

        y = torch.cat(ys, dim=1)
        time_logging.end("batch norm", time)
        return y


def test_batchnorm():
    bn = SE3BatchNorm([(3, 1), (4, 3), (1, 5)])

    x = torch.autograd.Variable(torch.randn(16, 3+12+5, 10, 10, 10))

    y = bn(x)
