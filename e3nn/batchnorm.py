# pylint: disable=C,R,E1101
import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1, affine=True, reduce='mean'):
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

        self.reduce = reduce

    def __repr__(self):
        return "{} (Rs={}, eps={}, momentum={})".format(
            self.__class__.__name__,
            self.Rs,
            self.eps,
            self.momentum)

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, stacked feature, x, y, z]
        '''

        if self.training:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0
        for m, d in self.Rs:
            field = input[:, ix: ix + m * d]  # [batch, feature * repr, x, y, z]
            ix += m * d

            # [batch, feature, repr, x * y * z]
            field = field.contiguous().view(input.size(0), m, d, -1)

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(0).mean(-1).view(-1)  # [feature]
                    new_means.append(
                        self._roll_avg(self.running_mean[irm:irm + m], field_mean)
                    )
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m

                # [batch, feature, repr, x * y * z]
                field = field - field_mean.view(1, m, 1, 1)

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                if self.reduce == 'mean':
                    field_norm = field_norm.mean(-1)  # [batch, feature]
                elif self.reduce == 'max':
                    field_norm = field_norm.max(-1)[0]  # [batch, feature]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                field_norm = field_norm.mean(0)  # [feature]
                new_vars.append(self._roll_avg(self.running_var[irv: irv + m], field_norm))
            else:
                field_norm = self.running_var[irv: irv + m]
            irv += m

            # [batch, feature, repr, x * y * z]
            field_norm = (field_norm + self.eps).pow(-0.5).view(1, m, 1, 1)

            if self.affine:
                weight = self.weight[iw: iw + m]  # [feature]
                iw += m

                # [batch, feature, repr, x * y * z]
                field_norm = field_norm * weight.view(1, m, 1, 1)

            field = field * field_norm  # [batch, feature, repr, x * y * z]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + m]  # [feature]
                ib += m
                field += bias.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            fields.append(field.view(input.size(0), m * d, *input.size()[2:]))

        if ix != input.size(1):
            fmt = "`ix` should have reached input.size(1) ({}), but it ended at {}"
            msg = fmt.format(input.size(1), ix)
            raise AssertionError(msg)

        if self.training:
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        if self.training:
            self.running_mean.copy_(torch.cat(new_means))
            self.running_var.copy_(torch.cat(new_vars))

        return torch.cat(fields, dim=1)  # [batch, stacked feature, x, y, z]
