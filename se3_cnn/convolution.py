# pylint: disable=C,R,E1101,E1102
import numpy as np
import torch
from se3_cnn import basis_kernels


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=basis_kernels.gaussian_window_fct_convenience_wrapper, verbose=True, **kwargs):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        multiplicity is a positive integer
        representation is a function of SO(3) in Euler ZYZ parametrisation alpha, beta, gamma

        :param upsampling: the sampling of the kernel is made on a grid `upsampling` time bigger and then subsampled with a gaussian
        '''
        super().__init__()

        self.size = size
        self.Rs_out = [(m, l) for m, l in Rs_out if m >= 1]
        self.Rs_in = [(m, l) for m, l in Rs_in if m >= 1]
        self.multiplicities_out = [m for m, _ in self.Rs_out]
        self.multiplicities_in = [m for m, _ in self.Rs_in]
        self.dims_out = [2 * l + 1 for _, l in self.Rs_out]
        self.dims_in = [2 * l + 1 for _, l in self.Rs_in]
        self.n_out = sum([self.multiplicities_out[i] * self.dims_out[i] for i in range(len(self.multiplicities_out))])
        self.n_in = sum([self.multiplicities_in[j] * self.dims_in[j] for j in range(len(self.multiplicities_in))])

        self.nweights = 0

        for i, (m_out, l_out) in enumerate(self.Rs_out):
            for j, (m_in, l_in) in enumerate(self.Rs_in):
                basis = basis_kernels.cube_basis_kernels_analytical(size, l_in, l_out, radial_window)
                if basis is not None:
                    assert basis.shape[1:] == ((2 * l_out + 1), (2 * l_in + 1), size, size, size), "wrong basis shape - your cache files may probably be corrupted"
                    # rescale each basis element such that the weight can be initialized with Normal(0,1)
                    # orthonormalization already done in cube_basis_kernels_analytical!
                    # ORIGINAL
                    # basis *= np.sqrt((2 * l_out + 1) / (len(basis)*m_in))
                    # EQUAL CONTRIB OF ALL SUPERBLOCKS
                    # orig normalized for one superblock of nmultiplicities_in capsules, disregarded that there are multiple in-orders -> divide by number of in-orders
                    # basis *= np.sqrt((2 * l_out + 1) / (len(basis)*m_in*len(Rs_in)))
                    # EQUAL CONTRIB OF ALL CAPSULES
                    basis *= np.sqrt((2 * l_out + 1) / (len(basis) * sum(self.multiplicities_in)))
                    if verbose:
                        N_sample = 5
                        overlaps = np.mean([basis_kernels.check_basis_equivariance(basis, l_in, l_out, a, b, c) for a, b, c in 2 * np.pi * np.random.rand(N_sample, 3)], axis=0)
                        print("{} -> {} : Created {} basis elements with equivariance {}".format(l_in, l_out, len(basis), overlaps))
                    self.register_buffer("kernel_{}_{}".format(i, j), torch.tensor(basis, dtype=torch.float))
                    self.nweights += m_out * m_in * basis.shape[0]
                else:
                    self.register_buffer("kernel_{}_{}".format(i, j), None)

        self.weight = torch.nn.Parameter(torch.randn(self.nweights))
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out}, size={size}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
            size=self.size,
            kwargs=self.kwargs
        )

    def combination(self, weight):
        kernel = weight.new_empty(self.n_out, self.n_in, self.size, self.size, self.size)

        weight_index = 0

        begin_i = 0
        for i, mi in enumerate(self.multiplicities_out):
            begin_j = 0
            for j, mj in enumerate(self.multiplicities_in):
                si = slice(begin_i, begin_i + mi * self.dims_out[i])
                sj = slice(begin_j, begin_j + mj * self.dims_in[j])

                kij = getattr(self, "kernel_{}_{}".format(i, j))
                if kij is not None:
                    b_el = kij.size(0)
                    b_size = kij.size()[1:]

                    w = weight[weight_index: weight_index + mi * mj * b_el].view(mi * mj, b_el)  # [I*J, beta]
                    weight_index += mi * mj * b_el

                    basis_kernels_ij = kij.contiguous().view(b_el, -1)  # [beta, i*j*x*y*z]

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

    def forward(self, input):  # pylint: disable=W
        kernel = self.combination(self.weight)

        output = torch.nn.functional.conv3d(input, kernel, **self.kwargs)

        return output


def test_normalization(batch, input_size, Rs_in, Rs_out, size):
    conv = SE3Convolution(Rs_in, Rs_out, size)

    print("Weights Number = {} Mean = {:.3f} Std = {:.3f}".format(conv.weight.numel(), conv.weight.data.mean(), conv.weight.data.std()))

    n_out = sum([m * (2 * l + 1) for m, l in Rs_out])
    n_in = sum([m * (2 * l + 1) for m, l in Rs_in])

    x = torch.randn(batch, n_in, input_size, input_size, input_size)
    print("x Number = {} Mean = {:.3f} Std = {:.3f}".format(x.numel(), x.data.mean(), x.data.std()))
    y = conv(x)

    assert y.size(1) == n_out

    print("y Number = {} Mean = {:.3f} Std = {:.3f}".format(y.numel(), y.data.mean(), y.data.std()))
    return y.data


def test_combination_gradient(Rs_in, Rs_out, size):
    conv = SE3Convolution(Rs_in, Rs_out, size, basis_kernels.gaussian_window_fct_convenience_wrapper, True)

    x = torch.rand(1, sum(m * (2 * l + 1) for m, l in Rs_in), 6, 6, 6, requires_grad=True)

    torch.autograd.gradcheck(conv, (x, ), eps=1)


def main():
    Rs_in = [(2, 0), (1, 1)]
    Rs_out = [(2, 0), (2, 1), (1, 2)]
    test_normalization(3, 15, Rs_in, Rs_out, 5)
    test_combination_gradient([(1, 0), (1, 1)], [(1, 0)], 5)


if __name__ == "__main__":
    main()
