# pylint: disable=C,R,E1101
import torch
import numpy as np


class NormActivation(torch.nn.Module):
    def __init__(self, dimensionalities, tensor_act=None, scalar_act=None, eps=1e-6, bias_min=.5, bias_max=2):
        '''
        :param dimensionalities: list of dimensionalities of the capsules
        :param scalar_act: activation function applied to scalar capsules - in last layer often set to None
        :param eps: regularazier added to norm to prevent division by zero
        :param bias_min: lower cutoff of uniform bias initialization
        :param bias_max: upper cutoff of uniform bias initialization

        scalar capsules are acted on by a ReLU nonlinearity, higher order capsules with a nonlinearity acting on their norm
        '''
        super().__init__()

        self.dimensionalities = dimensionalities
        self.tensor_act = torch.nn.Softplus(beta=1, threshold=20) if not tensor_act else tensor_act
        self.scalar_act = scalar_act
        self.is_scalar = [dim == 1 for dim in dimensionalities]
        nbias = int(np.sum(np.array(dimensionalities) != 1))
        self.bias = torch.nn.Parameter(torch.Tensor(nbias)) if nbias > 0 else None
        self.eps = eps
        self.bias_min = bias_min
        self.bias_max = bias_max
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.uniform_(self.bias_min, self.bias_max)

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        capsule_activations = []
        idx_capsule_begin = 0
        idx_bias = 0

        for dim, scalar_bool in zip(self.dimensionalities, self.is_scalar):
            # take capsule out of input
            capsule = input[:, idx_capsule_begin:idx_capsule_begin+dim]
            # act on scalar capsules with scalar activation
            if scalar_bool:
                if self.scalar_act == None:
                    capsule_activ = capsule
                else:
                    capsule_activ = self.scalar_act(capsule)
            # act on norms of higher order capsules
            else:
                norm = torch.norm(capsule, p=2, dim=1, keepdim=True) + self.eps  # [batch, 1, x, y, z]
                b = self.bias[idx_bias].expand_as(norm)  # [batch, 1, x, y, z]
                activ_factor = self.tensor_act(norm - b)  # [batch, 1, x, y, z]
                # activ_factor = 1 + torch.nn.ELU(norm - b.expand_as(norm)) # add 1 to make scaling factor positive
                capsule_activ = activ_factor * (capsule/norm)
                idx_bias += 1
            # append to list of nonlinearly transformed capsules
            capsule_activations.append(capsule_activ)
            idx_capsule_begin += dim
        assert idx_capsule_begin == input.size(1)
        if self.bias is not None:
            assert idx_bias == self.bias.size(0)
        return torch.cat(capsule_activations, dim=1)


class NormSoftplus(torch.nn.Module):
    def __init__(self, dimensionalities, scalar_act, eps=1e-6, bias_min=.5, bias_max=2):
        '''
        :param dimensionalities: list of dimensionalities of the capsules
        :param scalar_act: activation function applied to scalar capsules - in last layer often set to None
        :param eps: regularazier added to norm to prevent division by zero
        :param bias_min: lower cutoff of uniform bias initialization
        :param bias_max: upper cutoff of uniform bias initialization

        scalar capsules are acted on by a ReLU nonlinearity, higher order capsules with a nonlinearity acting on their norm
        '''
        super().__init__()

        self.dimensionalities = dimensionalities
        self.scalar_act = scalar_act
        self.is_scalar = [dim == 1 for dim in dimensionalities]
        nbias = int(np.sum(np.array(dimensionalities) != 1))
        self.bias = torch.nn.Parameter(torch.Tensor(nbias)) if nbias > 0 else None
        self.eps = eps
        self.bias_min = bias_min
        self.bias_max = bias_max
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.uniform_(self.bias_min, self.bias_max)

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        capsule_activations = []
        idx_capsule_begin = 0
        idx_bias = 0

        for dim, scalar_bool in zip(self.dimensionalities, self.is_scalar):
            # take capsule out of input
            capsule = input[:, idx_capsule_begin:idx_capsule_begin+dim]
            # act on scalar capsules with scalar activation
            if scalar_bool:
                if self.scalar_act == None:
                    capsule_activ = capsule
                else:
                    capsule_activ = self.scalar_act(capsule)
            # act on norms of higher order capsules
            else:
                norm = torch.norm(capsule, p=2, dim=1, keepdim=True) + self.eps  # [batch, 1, x, y, z]
                b = self.bias[idx_bias].expand_as(norm)  # [batch, 1, x, y, z]
                activ_factor = torch.nn.Softplus(beta=1, threshold=20)(norm - b)  # [batch, 1, x, y, z]
                # activ_factor = 1 + torch.nn.ELU(norm - b.expand_as(norm)) # add 1 to make scaling factor positive
                capsule_activ = activ_factor * (capsule/norm)
                idx_bias += 1
            # append to list of nonlinearly transformed capsules
            capsule_activations.append(capsule_activ)
            idx_capsule_begin += dim
        assert idx_capsule_begin == input.size(1)
        if self.bias is not None:
            assert idx_bias == self.bias.size(0)
        return torch.cat(capsule_activations, dim=1)


class NormRelu(torch.nn.Module):
    def __init__(self, enable):
        '''
        :param enable: list of tuple (dimension, boolean)

        If boolean is True a bias and relu will be applied
        '''
        super().__init__()

        self.enable = enable
        nbias = sum([1 for d, on in self.enable if on])
        self.bias = torch.nn.Parameter(torch.FloatTensor(nbias)) if nbias > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data[:] = 0.1

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, feature, x, y, z]
        '''
        if self.bias is None:
            return input

        xs = []
        begin1 = 0
        begin2 = 0

        for d, on in self.enable:
            x = input[:, begin1:begin1 + d]

            if on:
                x = NormReluFunction()(x, self.bias[begin2:begin2 + 1])

                begin2 += 1

            xs.append(x)

            begin1 += d

        assert begin1 == input.size(1)
        assert begin2 == self.bias.size(0)

        return torch.cat(xs, dim=1)


class NormReluFunction(torch.autograd.Function):
    def forward(self, x, b):  # pylint: disable=W
        norm = torch.sqrt(torch.sum(x * x, dim=1)) + 1e-8  # [batch, x, y, z]
        newnorm = norm - b.expand_as(norm)  # [batch, x, y, z]
        newnorm[newnorm < 0] = 0
        ratio = newnorm / norm
        ratio = ratio.view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x)

        self.save_for_backward(x, b)
        r = x * ratio
        return r

    def backward(self, grad_out):  # pylint: disable=W
        x, b = self.saved_tensors

        norm = torch.sqrt(torch.sum(x * x, dim=1)) + 1e-8  # [batch, x, y, z]

        grad_x = grad_b = None

        if self.needs_input_grad[0]:
            newnorm = norm - b.expand_as(norm)  # [batch, x, y, z]
            newnorm[newnorm < 0] = 0
            ratio = newnorm / norm
            ratio = ratio.view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x)

            grad_x = grad_out * ratio
            grad_x += torch.sum(grad_out * x, dim=1, keepdim=True).expand_as(x) * x / \
                (norm ** 2).view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x) * (1 - ratio)
            grad_x[ratio <= 0] = 0

        if self.needs_input_grad[1]:
            grad_b = -torch.sum(grad_out * x, dim=1) / norm
            grad_b[norm < b] = 0
            grad_b = torch.sum(grad_b.view(-1), dim=0)

        return grad_x, grad_b


def test_norm_relu_gradient():
    x = torch.autograd.Variable(torch.rand(1, 5, 3, 3, 3), requires_grad=True)
    b = torch.autograd.Variable(torch.rand(1), requires_grad=True)
    torch.autograd.gradcheck(NormReluFunction(), (x, b), eps=1e-3, rtol=1e-2)
