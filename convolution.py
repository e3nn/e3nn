#pylint: disable=C,R,E1101
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import basis_kernels

class SE3Convolution(torch.nn.Module):
    def __init__(self, size, Rs_out, Rs_in):
        '''
        Rs_out = [(representation, multiplicity), ...]
        '''
        super(SE3Convolution, self).__init__()
        M = 15

        #TODO for now, only one parameter radially
        rng = np.linspace(start=-1, stop=1, num=size * M, endpoint=True)
        z, y, x = np.meshgrid(rng, rng, rng)
        r = np.sqrt(x**2 + y**2 + z**2)
        mask = np.cos((r - 0.5) * 2 * np.pi) + 1
        mask[r > 1] = 0

        self.size = size
        self.multiplicites_out = [x[1] for x in Rs_out]
        self.multiplicites_in = [x[1] for x in Rs_in]
        self.kernels = [
            [torch.FloatTensor(
                basis_kernels.gaussian_subsampling(
                    basis_kernels.cube_basis_kernels(size * M, R_out, R_in) * mask,
                    (1, 1, 1, M, M, M)))
            for R_in, _ in Rs_in]
            for R_out, _ in Rs_out]
        self.dims_out = [basis_kernels.dim(R) for R, _ in Rs_out]
        self.dims_in = [basis_kernels.dim(R) for R, _ in Rs_in]

        nweights = 0
        for i in range(len(Rs_out)):
            for j in range(len(Rs_in)):
                nweights += self.multiplicites_out[i] * self.multiplicites_in[j] * self.kernels[i][j].size(0)

        self.weight = Parameter(torch.Tensor(nweights))

    def reset_parameters(self):
        self.weight.normal_(0, 1)

    def forward(self, x):

        cv = _SE3Convolution(self.kernels, self.multiplicites_out, self.multiplicites_in, self.dims_out, self.dims_in, self.size)

        return cv(x, self.weight)


class _SE3Convolution(torch.autograd.Function):
    def __init__(self, kernels, multiplicites_out, multiplicites_in, dims_out, dims_in, size):
        super(_SE3Convolution, self).__init__()
        self.kernels = kernels
        self.multiplicites_out = multiplicites_out
        self.multiplicites_in = multiplicites_in
        self.dims_out = dims_out
        self.dims_in = dims_in
        self.size = size

    def forward(self, x, w):
        n_out = sum([self.multiplicites_out[i] * self.dims_out[i] for i in range(len(self.multiplicites_out))])
        n_in = sum([self.multiplicites_in[j] * self.dims_in[j] for j in range(len(self.multiplicites_in))])

        assert x.size(1) == n_in

        weight = torch.FloatTensor(n_out, n_in, self.size, self.size, self.size)

        weight_index = 0

        begin_i = 0
        for i in range(len(self.multiplicites_out)):
            begin_j = 0
            for j in range(len(self.multiplicites_in)):
                for ii in range(self.multiplicites_out[i]):
                    si = slice(begin_i + ii * self.dims_out[i], begin_i + (ii + 1) * self.dims_out[i])
                    for jj in range(self.multiplicites_in[j]):
                        sj = slice(begin_j + jj * self.dims_in[j], begin_j + (jj + 1) * self.dims_in[j])
                        for k in range(self.kernels[i][j].size(0)):
                            weight[si, sj] = self.kernels[i][j][k] * w[weight_index]
                            weight_index += 1
                begin_j += self.multiplicites_in[j] * self.dims_in[j]
            begin_i += self.multiplicites_out[i] * self.dims_out[i]

        return conv3d(x, weight)

    def backward(self, grad_output): #pylint: disable=W
        return None, None


def conv3d(x, w):
    x = Variable(x)
    w = Variable(w)
    r = torch.nn.functional.conv3d(x, w)
    return r.data
