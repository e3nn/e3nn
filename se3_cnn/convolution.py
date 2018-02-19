# pylint: disable=C,R,E1101
import numpy as np
import torch
from se3_cnn import basis_kernels
from se3_cnn import SO3


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, verbose=True, **kwargs):
        '''
        :param Rs_in: list of couple (multiplicity, representation)
        :param Rs_out: list of couple (multiplicity, representation)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma

        :param upsampling: the sampling of the kernel is made on a grid `upsampling` time bigger and then subsampled with a gaussian
        '''
        super().__init__()

        self.combination = SE3KernelCombination(Rs_in, Rs_out, size, radial_window_dict, verbose)
        self.weight = torch.nn.Parameter(torch.randn(self.combination.nweights))
        self.kwargs = kwargs

    def __repr__(self):
        return "{} (size={}, {})".format(
            self.__class__.__name__,
            self.combination.size,
            self.kwargs)

    def forward(self, input):  # pylint: disable=W
        kernel = self.combination(self.weight)

        output = torch.nn.functional.conv3d(input, kernel, **self.kwargs)

        return output


class SE3KernelCombination(torch.autograd.Function):
    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, verbose):
        super().__init__()

        self.size = size
        Rs_out = [(m, R) for m, R in Rs_out if m >= 1]
        Rs_in = [(m, R) for m, R in Rs_in if m >= 1]
        self.multiplicities_out = [m for m, _ in Rs_out]
        self.multiplicities_in = [m for m, _ in Rs_in]
        self.dims_out = [SO3.dim(R) for _, R in Rs_out]
        self.dims_in = [SO3.dim(R) for _, R in Rs_in]
        self.n_out = sum([self.multiplicities_out[i] * self.dims_out[i] for i in range(len(self.multiplicities_out))])
        self.n_in = sum([self.multiplicities_in[j] * self.dims_in[j] for j in range(len(self.multiplicities_in))])

        self.kernels = []
        self.nweights = 0

        for m_out, R_out in Rs_out:
            self.kernels.append([])
            for m_in, R_in in Rs_in:
                basis = self._generate_basis(R_in, R_out, m_in, m_out, size, radial_window_dict, verbose)








                # ORIGINAL
                # basis *= np.sqrt(SO3.dim(R_out) / (len(basis)*m_in))

                # EQUAL CONTRIB OF ALL SUPERBLOCKS
                # orig normalized for one superblock of nmultiplicities_in capsules, disregarded that there are multiple in-orders -> divide by number of in-orders
                # basis *= np.sqrt(SO3.dim(R_out) / (len(basis)*m_in*len(Rs_in)))

                # EQUAL CONTRIB OF ALL CAPSULES
                # more sensible?
                basis *= np.sqrt(SO3.dim(R_out) / (len(basis)*sum(self.multiplicities_in)))







                if basis is not None:
                    self.kernels[-1].append(torch.FloatTensor(basis))
                    self.nweights += m_out * m_in * basis.shape[0]
                else:
                    self.kernels[-1].append(None)







    def _generate_basis(self, R_in, R_out, m_in, m_out, size, radial_window_dict, verbose):  # pylint: disable=W0613
        basis = basis_kernels.cube_basis_kernels_analytical(size, R_in, R_out, radial_window_dict)
        assert basis.shape[1:] == (SO3.dim(R_out), SO3.dim(R_in), size, size, size), "wrong basis shape - your cache files may probably be corrupted"







        # rescale each basis element such that the weight can be initialized with Normal(0,1)
        # orthonormalization already done in cube_basis_kernels_analytical!
        for k in range(len(basis)):
            basis[k] *= np.sqrt(SO3.dim(R_out) / len(basis))
            basis[k] /= np.sqrt(m_in)





        if verbose:
            N_sample = 5
            overlaps = np.mean([basis_kernels.check_basis_equivariance(basis, R_out, R_in, a,b,c) for a,b,c in 2*np.pi*np.random.rand(N_sample,3)], axis=0)
            print("{} -> {} : Created {} basis elements with equivariance {}".format(R_in.__name__, R_out.__name__, len(basis), overlaps))

        return basis if basis.shape[0] > 0 else None

    def _cuda_kernels(self, cuda):
        for row in self.kernels:
            for i in range(len(row)):
                ker = row[i]
                if ker is None:
                    pass
                elif ker.is_cuda and not cuda:
                    ker = ker.cpu()
                elif not ker.is_cuda and cuda:
                    ker = ker.cuda()

                row[i] = ker

    def forward(self, weight):  # pylint: disable=W
        """
        :return: [feature_out, feature_in, x, y, z]
        """
        assert weight.dim() == 1, "size = {}".format(weight.size())
        assert weight.size(0) == self.nweights

        self._cuda_kernels(weight.is_cuda)

        if weight.is_cuda:
            kernel = torch.cuda.FloatTensor(self.n_out, self.n_in, self.size, self.size, self.size)
        else:
            kernel = torch.FloatTensor(self.n_out, self.n_in, self.size, self.size, self.size)

        weight_index = 0

        begin_i = 0
        for i, mi in enumerate(self.multiplicities_out):
            begin_j = 0
            for j, mj in enumerate(self.multiplicities_in):
                si = slice(begin_i, begin_i + mi * self.dims_out[i])
                sj = slice(begin_j, begin_j + mj * self.dims_in[j])

                if self.kernels[i][j] is not None:
                    b_el = self.kernels[i][j].size(0)
                    b_size = self.kernels[i][j].size()[1:]

                    w = weight[weight_index: weight_index + mi * mj * b_el].view(mi * mj, b_el)  # [I*J, beta]
                    weight_index += mi * mj * b_el

                    basis_kernels_ij = self.kernels[i][j].view(b_el, -1)  # [beta, i*j*x*y*z]

                    ker = torch.mm(w, basis_kernels_ij)  # [I*J, i*j*x*y*z]
                    ker = ker.view(mi, mj, *b_size)  # [I, J, i, j, x, y, z]
                    ker = ker.transpose(1, 2).contiguous()  # [I, i, J, j, x, y, z]
                    ker = ker.view(mi * self.dims_out[i], mj * self.dims_in[j], *b_size[2:])  # [I*i, J*j, x, y, z]
                    kernel[si, sj] = ker
                else:
                    kernel[si, sj] = 0

                begin_j += mj * self.dims_in[j]
            begin_i += mi * self.dims_out[i]

        return kernel

    def backward(self, grad_kernel):  # pylint: disable=W
        self._cuda_kernels(grad_kernel.is_cuda)

        if grad_kernel.is_cuda:
            grad_weight = torch.cuda.FloatTensor(self.nweights)
        else:
            grad_weight = torch.FloatTensor(self.nweights)

        weight_index = 0

        begin_i = 0
        for i, mi in enumerate(self.multiplicities_out):
            begin_j = 0
            for j, mj in enumerate(self.multiplicities_in):
                if self.kernels[i][j] is not None:
                    b_el = self.kernels[i][j].size(0)
                    basis_kernels_ij = self.kernels[i][j]  # [beta, i, j, x, y, z]
                    basis_kernels_ij = basis_kernels_ij.view(b_el, -1)  # [beta, i*j*x*y*z]

                    si = slice(begin_i, begin_i + mi * self.dims_out[i])
                    sj = slice(begin_j, begin_j + mj * self.dims_in[j])

                    grad = grad_kernel[si, sj]  # [I * i, J * j, x, y, z]
                    grad = grad.contiguous().view(mi, self.dims_out[i], mj, self.dims_in[j], -1).transpose(1, 2)  # [I, J, i, j, x*y*z]
                    grad = grad.contiguous().view(mi * mj, -1)  # [I*J, i*j*x*y*z]
                    grad = torch.mm(grad, basis_kernels_ij.transpose(0, 1))  # [I*J, beta]

                    grad_weight[weight_index: weight_index + mi * mj * b_el] = grad.view(-1)  # [I * J * beta]
                    weight_index += mi * mj * b_el

                begin_j += self.multiplicities_in[j] * self.dims_in[j]
            begin_i += self.multiplicities_out[i] * self.dims_out[i]

        return grad_weight


def test_normalization(batch, size, Rs_out=None, Rs_in=None):
    if Rs_out is None:
        Rs_out = [(1, SO3.repr1), (1, SO3.repr3)]
    if Rs_in is None:
        Rs_in = [(1, SO3.repr1), (1, SO3.repr3)]
    conv = SE3Convolution(4, 2, Rs_in, Rs_out)
    print("Weights Amount = {} Mean = {} Std = {}".format(conv.weight.numel(), conv.weight.data.mean(), conv.weight.data.std()))

    n_out = sum([m * SO3.dim(r) for m, r in Rs_out])
    n_in = sum([m * SO3.dim(r) for m, r in Rs_in])

    x = torch.autograd.Variable(torch.randn(batch, n_in, size, size, size))
    print("x Amount = {} Mean = {} Std = {}".format(x.numel(), x.data.mean(), x.data.std()))
    y = conv(x)

    assert y.size(1) == n_out

    print("y Amount = {} Mean = {} Std = {}".format(y.numel(), y.data.mean(), y.data.std()))
    return y.data


def test_combination_gradient(size, n_radial, upsamplig, Rs_out=None, Rs_in=None, **kargs):
    if Rs_in is None:
        Rs_in = [(2, SO3.repr1), (1, SO3.repr3), (1, SO3.repr5)]
    if Rs_out is None:
        Rs_out = [(1, SO3.repr3), (1, SO3.repr1)]

    combination = SE3KernelCombination(size, n_radial, upsamplig, Rs_out, Rs_in, **kargs)

    w = torch.autograd.Variable(torch.rand(combination.nweights), requires_grad=True)

    torch.autograd.gradcheck(combination, (w, ), eps=1)
