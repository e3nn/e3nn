# pylint: disable=C,R,E1101
import torch
import torch.nn as nn


class SE3BatchNorm(nn.Module):
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


    def _roll_avg(self, x, y):
        return (1 - self.momentum) * x + self.momentum * y


    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, stacked feature, x, y, z]
        '''

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
            field = field.contiguous().view(input.size(0), m, d, -1)  # [batch, feature, repr, x * y * z]

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(0).mean(-1).view(-1).detach()  # [feature]
                    new_means.append(self._roll_avg(self.running_mean[irm:irm+m], field_mean))
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m
                field = field - field_mean.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                if self.reduce == 'mean':
                    field_norm = field_norm.mean(-1)  # [batch, feature]
                elif self.reduce == 'max':
                    field_norm = field_norm.max(-1)[0]  # [batch, feature]
                else:
                    raise ValueError("Invalid reduce option")
                field_norm = field_norm.mean(0).detach()  # [feature]
                new_vars.append(self._roll_avg(self.running_var[irv: irv+m], field_norm))
            else:
                field_norm = self.running_var[irv: irv + m]
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

        self.running_mean = torch.cat(new_means) 
        self.running_var = torch.cat(new_vars)

        return torch.cat(fields, dim=1)  # [batch, stacked feature, x, y, z]


from se3cnn import SE3Kernel
from se3cnn import kernel


class SE3BNConvolution(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3BatchNorm followed by SE3Convolution
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window=kernel.gaussian_window_wrapper, dyn_iso=False, verbose=False, eps=1e-5, momentum=0.1, reduce='mean', **kwargs):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.kernel = SE3Kernel(Rs_in, Rs_out, size, radial_window, dyn_iso, verbose)
        self.kwargs = kwargs

        self.Rs = list(zip(self.kernel.multiplicities_in, self.kernel.dims_in))
        num_scalar = sum(m for m, d in self.Rs if d == 1)
        num_features = sum(m for m, d in self.Rs)

        self.register_buffer('running_mean', torch.zeros(num_scalar))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.reduce = reduce

    def __repr__(self):
        return "{name} ({ker}, eps={eps}, momentum={momentum})".format(
            name=self.__class__.__name__,
            ker=self.kernel,
            **self.__dict__,
            **self.kwargs)


    def _roll_avg(self, x, y):
        return (1 - self.momentum) * x + self.momentum * y


    def forward(self, input):  # pylint: disable=W
        field_means = []
        field_norms = []
        
        new_means = []
        new_vars = []

        ix = 0
        irm = 0
        irv = 0
        for m, d in self.Rs:
            field = input[:, ix: ix + m * d]  # [batch, feature * repr, x, y, z]
            ix += m * d
            field = field.contiguous().view(input.size(0), m, d, -1)  # [batch, feature, repr, x * y * z]

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(-1).mean(0).view(-1).detach()  # [feature]
                    new_means.append(self._roll_avg(self.running_mean[irm: irm+m], field_mean))
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m
                field = field - field_mean.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]
                field_means.append(field_mean)  # [feature]

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                if self.reduce == 'mean':
                    field_norm = field_norm.mean(-1)  # [batch, feature]
                elif self.reduce == 'max':
                    field_norm = field_norm.max(-1)[0]  # [batch, feature]
                else:
                    raise ValueError("Invalid reduce option")
                field_norm = field_norm.mean(0).detach()  # [feature]
                new_vars.append(self._roll_avg(self.running_var[irv: irv+m], field_norm))
            else:
                field_norm = self.running_var[irv: irv + m]
            irv += m

            field_norm = (field_norm + self.eps).pow(-0.5)  # [feature]
            field_norms.append(field_norm)  # [feature]
            del field

        if ix != input.size(1):
            fmt = "`ix` should have reached input.size(1) ({}), but it ended at {}"
            msg = fmt.format(input.size(1), ix)
            raise AssertionError(msg)

        assert irm == self.running_mean.numel()
        assert irv == self.running_var.size(0)

        self.running_mean = torch.cat(new_means)
        self.running_var = torch.cat(new_vars)

        bias = []

        ws = []
        weight_index = 0
        for i, (mi, di) in enumerate(zip(self.kernel.multiplicities_out, self.kernel.dims_out)):
            index_mean = 0
            bia = input.new_zeros(mi * di)
            for j, (mj, dj, normj) in enumerate(zip(self.kernel.multiplicities_in, self.kernel.dims_in, field_norms)):
                kernel = getattr(self.kernel, "kernel_{}_{}".format(i, j))
                if kernel is not None:
                    b_el = kernel.size(0)

                    w = self.kernel.weight[weight_index: weight_index + mi * mj * b_el]
                    weight_index += mi * mj * b_el

                    w = w.view(mi, mj, b_el) * normj.view(1, -1, 1)  # [feature_out, feature_in, basis]
                    ws.append(w.view(-1))

                    if di == 1 and dj == 1:
                        mean = field_means[index_mean]  # [feature_in]
                        index_mean += 1

                        identity = kernel.view(b_el, -1).sum(-1)  # [basis]
                        bia -= torch.mm(torch.mm(w.view(-1, b_el), identity.view(b_el, 1)).view(mi, mj), mean.view(-1, 1)).view(-1)  # [feature_out]
            bias.append(bia)

        bias = torch.cat(bias)
        kernel = self.kernel.combination(torch.cat(ws))
        return torch.nn.functional.conv3d(input, kernel, bias=bias, **self.kwargs)
