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
                    field_mean = self.running_mean[irm: irm + m]
                irm += m
                field = field - field_mean.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                field_norm = field_norm.mean(0).mean(-1)  # [feature]
                self.running_var[irv: irv + m] = (1 - self.momentum) * self.running_var[irv: irv + m] + self.momentum * field_norm.data
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
                field = field + bias.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]

            fields.append(field.view(input.size(0), m * d, *input.size()[2:]))

        assert ix == input.size(1)
        if self.training:
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        return torch.cat(fields, dim=1)  # [batch, stacked feature, x, y, z]


def test_batchnorm(Rs=None):
    if Rs is None:
        Rs = [(3, 1), (4, 3), (1, 5)]

    bn = SE3BatchNorm(Rs)

    x = torch.randn(16, sum(m * d for m, d in Rs), 10, 10, 10) * 3 + 12

    bn.train()
    y = bn(x)

    bn.eval()
    z = bn(x)

    return y, z


from se3_cnn.convolution import SE3Convolution
from se3_cnn import basis_kernels


class SE3BNConvolution(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3BatchNorm followed by SE3Convolution
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window=basis_kernels.gaussian_window_fct_convenience_wrapper, verbose=True, eps=1e-5, momentum=0.1, **kwargs):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.conv = SE3Convolution(Rs_in, Rs_out, size, radial_window, verbose)
        self.kwargs = kwargs

        self.Rs = list(zip(self.conv.multiplicities_in, self.conv.dims_in))
        num_scalar = sum(m for m, d in self.Rs if d == 1)
        num_features = sum(m for m, d in self.Rs)

        self.register_buffer('running_mean', torch.zeros(num_scalar))
        self.register_buffer('running_var', torch.ones(num_features))

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, size={size}, eps={eps}, momentum={momentum})".format(
            name=self.__class__.__name__,
            Rs_in=self.conv.Rs_in,
            Rs_out=self.conv.Rs_out,
            size=self.conv.size,
            **self.__dict__,
            **self.kwargs)

    def forward(self, input):  # pylint: disable=W
        field_means = []
        field_norms = []

        ix = 0
        irm = 0
        irv = 0
        for m, d in self.Rs:
            field = input[:, ix: ix + m * d]  # [batch, feature * repr, x, y, z]
            ix += m * d
            field = field.contiguous().view(input.size(0), m, d, -1)  # [batch, feature, repr, x * y * z]

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(0).mean(-1).view(-1)  # [feature]
                    self.running_mean[irm: irm + m] = (1 - self.momentum) * self.running_mean[irm: irm + m] + self.momentum * field_mean.data
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m
                field = field - field_mean.view(1, m, 1, 1)  # [batch, feature, repr, x * y * z]
                field_means.append(field_mean)  # [feature]

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [batch, feature, x * y * z]
                field_norm = field_norm.mean(0).mean(-1)  # [feature]
                self.running_var[irv: irv + m] = (1 - self.momentum) * self.running_var[irv: irv + m] + self.momentum * field_norm.data
            else:
                field_norm = self.running_var[irv: irv + m]
            irv += m

            field_norm = (field_norm + self.eps).pow(-0.5)  # [feature]
            field_norms.append(field_norm)  # [feature]

        assert ix == input.size(1)
        assert irm == self.running_mean.numel()
        assert irv == self.running_var.size(0)

        bias = []

        ws = []
        weight_index = 0
        for i, (mi, di) in enumerate(zip(self.conv.multiplicities_out, self.conv.dims_out)):
            index_mean = 0
            bia = input.data.new_zeros(mi * di)
            for j, (mj, dj, normj) in enumerate(zip(self.conv.multiplicities_in, self.conv.dims_in, field_norms)):
                kernel = getattr(self.conv, "kernel_{}_{}".format(i, j))
                if kernel is not None:
                    b_el = kernel.size(0)

                    w = self.conv.weight[weight_index: weight_index + mi * mj * b_el]
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
        kernel = self.conv.combination(torch.cat(ws))
        return torch.nn.functional.conv3d(input, kernel, bias=bias, **self.kwargs)


def test_bn_conv(Rs_in, Rs_out, kernel_size, batch, input_size):
    # input
    n_out = sum([m * (2 * l + 1) for m, l in Rs_out])
    n_in = sum([m * (2 * l + 1) for m, l in Rs_in])
    x = torch.rand(batch, n_in, input_size, input_size, input_size) * 2 + 2

    # BNConv
    bnconv = SE3BNConvolution(Rs_in, Rs_out, kernel_size)
    bnconv.train()
    y1 = bnconv(x).data

    assert y1.size(1) == n_out

    # BN + Conv
    bn = SE3BatchNorm(bnconv.Rs, affine=False)
    bn.train()

    conv = SE3Convolution(Rs_in, Rs_out, kernel_size)
    conv.train()
    conv.weight = bnconv.conv.weight

    y2 = conv(bn(x)).data

    assert y2.size(1) == n_out

    # compare
    assert (y2 - y1).std() / y2.std() < 1e-4


def main():
    test_batchnorm([(2, 1), (1, 3)])
    test_bn_conv([(2, 0), (2, 1), (1, 2), (1, 3)], [(2, 0), (2, 1), (1, 2)], 5, 4, 10)


if __name__ == "__main__":
    main()
