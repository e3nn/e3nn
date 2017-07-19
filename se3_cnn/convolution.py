#pylint: disable=C,R,E1101
import numpy as np
import torch
from torch.nn.parameter import Parameter
from se3_cnn import basis_kernels
from se3_cnn import SO3
from se3_cnn.non_linearities.scalar_activation import BiasRelu
from se3_cnn.non_linearities.norm_activation import NormRelu
from se3_cnn.utils import time_logging

class SE3Convolution(torch.nn.Module):
    def __init__(self, size, Rs_out, Rs_in, bias_relu=False, norm_relu=False, M=15, central_base=True):
        '''
        :param Rs_out: list of couple (multiplicity, representation)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma

        :param M: the sampling of the kernel is made on a grid M time bigger and then subsampled with a gaussian
        '''
        super(SE3Convolution, self).__init__()
        self.combination = SE3KernelCombination(size, Rs_out, Rs_in, M=M, central_base=central_base)
        self.weight = Parameter(torch.Tensor(self.combination.nweights))
        self.bias_relu = BiasRelu([(m * SO3.dim(R), SO3.dim(R) == 1) for m, R in Rs_out]) if bias_relu else None
        self.norm_relu = NormRelu([(SO3.dim(R), SO3.dim(R) > 1) for m, R in Rs_out for _ in range(m)]) if norm_relu else None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.bias_relu is not None:
            self.bias_relu.reset_parameters()
        if self.norm_relu is not None:
            self.norm_relu.reset_parameters()

    def forward(self, input): # pylint: disable=W
        time = time_logging.start()
        kernel = self.combination(self.weight)
        time = time_logging.end("kernel combination", time)
        output = torch.nn.functional.conv3d(input, kernel)
        time = time_logging.end("3d convolutions", time)
        if self.bias_relu is not None:
            output = self.bias_relu(output)
            time = time_logging.end("bias and relu", time)
        if self.norm_relu is not None:
            output = self.norm_relu(output)
            time = time_logging.end("norm relu", time)
        return output


class SE3KernelCombination(torch.autograd.Function):
    def __init__(self, size, Rs_out, Rs_in, M=15, central_base=True):
        super(SE3KernelCombination, self).__init__()

        self.size = size
        Rs_out = [(m, R) for m, R in Rs_out if m >= 1]
        Rs_in = [(m, R) for m, R in Rs_in if m >= 1]
        self.multiplicites_out = [m for m, _ in Rs_out]
        self.multiplicites_in = [m for m, _ in Rs_in]
        self.dims_out = [basis_kernels.dim(R) for _, R in Rs_out]
        self.dims_in = [basis_kernels.dim(R) for _, R in Rs_in]

        def generate_basis(R_out, R_in):
            basis = basis_kernels.cube_basis_kernels_subsampled_cosine(size, R_out, R_in, M)

            if central_base and size % 2 == 1:
                Ks = basis_kernels.basis_kernels_satisfying_SO3_constraint(R_out, R_in)
                center = np.zeros((len(Ks),) + basis.shape[1:])
                for k, K in enumerate(Ks):
                    center[k, :, :, size//2, size//2, size//2] = K
                basis = np.concatenate((center, basis))

            # normalize each basis element
            for k in range(len(basis)):
                basis[k] /= np.linalg.norm(basis[k])
                basis[k] *= np.sqrt(SO3.dim(R_out) / len(basis))
                # such that the weight can be initialized with Normal(0,1)
            return basis

        self.kernels = [[torch.FloatTensor(generate_basis(R_out, R_in))
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

        if weight.is_cuda:
            self.kernels = [[K.cuda() for K in row] for row in self.kernels]
            kernel = torch.cuda.FloatTensor(n_out, n_in, self.size, self.size, self.size)
        else:
            self.kernels = [[K.cpu() for K in row] for row in self.kernels]
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
        if grad_kernel.is_cuda:
            self.kernels = [[K.cuda() for K in row] for row in self.kernels]
            grad_weight = torch.cuda.FloatTensor(self.nweights)
        else:
            self.kernels = [[K.cpu() for K in row] for row in self.kernels]
            grad_weight = torch.FloatTensor(self.nweights)

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
                            grad_weight[weight_index] = torch.matmul(self.kernels[i][j][k].view(-1), grad_kernel[si, sj].contiguous().view(-1))
                            weight_index += 1
                begin_j += self.multiplicites_in[j] * self.dims_in[j]
            begin_i += self.multiplicites_out[i] * self.dims_out[i]

        return grad_weight
