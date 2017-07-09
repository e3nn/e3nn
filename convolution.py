#pylint: disable=C,R,E1101
import numpy as np
import torch
from torch.nn.parameter import Parameter
import basis_kernels

class SE3Convolution(torch.nn.Module):
    def __init__(self, size, Rs_out, Rs_in):
        '''
        :param Rs_out: list of couple (multiplicity, representation)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma
        '''
        super(SE3Convolution, self).__init__()
        self.combination = SE3KernelCombination(size, Rs_out, Rs_in)
        self.weight = Parameter(torch.Tensor(self.combination.nweights))

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input): # pylint: disable=W
        kernel = self.combination(self.weight)
        output = torch.nn.functional.conv3d(input, kernel)
        return output


class SE3KernelCombination(torch.autograd.Function):
    def __init__(self, size, Rs_out, Rs_in):
        super(SE3KernelCombination, self).__init__()
        M = 15

        self.size = size
        Rs_out = [(m, R) for m, R in Rs_out if m >= 1]
        Rs_in = [(m, R) for m, R in Rs_in if m >= 1]
        self.multiplicites_out = [m for m, _ in Rs_out]
        self.multiplicites_in = [m for m, _ in Rs_in]
        self.dims_out = [basis_kernels.dim(R) for _, R in Rs_out]
        self.dims_in = [basis_kernels.dim(R) for _, R in Rs_in]

        #TODO for now, only one parameter radially
        rng = np.linspace(start=-1, stop=1, num=size * M, endpoint=True)
        z, y, x = np.meshgrid(rng, rng, rng)
        r = np.sqrt(x**2 + y**2 + z**2)
        mask = np.cos((r - 0.5) * 2 * np.pi) + 1
        mask[r > 1] = 0

        self.kernels = [
            [torch.FloatTensor(
                basis_kernels.gaussian_subsampling(
                    basis_kernels.cube_basis_kernels(size * M, R_out, R_in) * mask,
                    (1, 1, 1, M, M, M)))
            for _, R_in in Rs_in]
            for _, R_out in Rs_out]

        self.nweights = 0
        for i in range(len(Rs_out)):
            for j in range(len(Rs_in)):
                self.nweights += self.multiplicites_out[i] * self.multiplicites_in[j] * self.kernels[i][j].size(0)

    def forward(self, weight): # pylint: disable=W
        assert weight.dim() == 1
        assert weight.size(0) == self.nweights
        n_out = sum([self.multiplicites_out[i] * self.dims_out[i] for i in range(len(self.multiplicites_out))])
        n_in = sum([self.multiplicites_in[j] * self.dims_in[j] for j in range(len(self.multiplicites_in))])

        kernel = torch.FloatTensor(n_out, n_in, self.size, self.size, self.size)

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
                            kernel[si, sj] = self.kernels[i][j][k] * weight[weight_index]
                            weight_index += 1
                begin_j += self.multiplicites_in[j] * self.dims_in[j]
            begin_i += self.multiplicites_out[i] * self.dims_out[i]

        return kernel

    def backward(self, grad_kernel): #pylint: disable=W
        grad_weight = torch.FloatTensor(self.nweights)

        weight_index = 0

        begin_i = 0
        for i in range(len(self.multiplicites_out)):
            begin_j = 0
            for j in range(len(self.multiplicites_in)):
                for ii in range(self.multiplicites_out[i]):
                    si = slice(begin_i + ii * self.dims_out[i], begin_i + (ii + 1) * self.dims_out[i])
                    for jj in range(self.multiplicites_in[j]):
                        print(i, j, ii, jj)
                        sj = slice(begin_j + jj * self.dims_in[j], begin_j + (jj + 1) * self.dims_in[j])
                        for k in range(self.kernels[i][j].size(0)):
                            grad_weight[weight_index] = torch.matmul(self.kernels[i][j][k].view(-1), grad_kernel[si, sj].contiguous().view(-1))
                            weight_index += 1
                begin_j += self.multiplicites_in[j] * self.dims_in[j]
            begin_i += self.multiplicites_out[i] * self.dims_out[i]

        return grad_weight
