#pylint: disable=C,R,E1101
import torch
from torch.nn.parameter import Parameter
from se3_cnn.utils import time_logging

class NormRelu(torch.nn.Module):
    def __init__(self, enable):
        '''
        :param enable: list of tuple (dimension, boolean)

        If boolean is True a bias and relu will be applied
        '''
        super(NormRelu, self).__init__()

        self.enable = enable
        nbias = sum([1 for d, on in self.enable if on])
        self.bias = Parameter(torch.FloatTensor(nbias)) if nbias > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data[:] = 0.1

    def forward(self, input): # pylint: disable=W
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
                x = NormReluFunction()(x, self.bias[begin2:begin2+1])

                begin2 += 1

            xs.append(x)

            begin1 += d

        assert begin1 == input.size(1)
        assert begin2 == self.bias.size(0)

        return torch.cat(xs, dim=1)


class NormReluFunction(torch.autograd.Function):
    def __init__(self):
        super(NormReluFunction, self).__init__()

    def forward(self, x, b): # pylint: disable=W
        time = time_logging.start()
        norm = torch.sqrt(torch.sum(x * x, dim=1)) + 1e-8 # [batch, x, y, z]
        newnorm = norm - b.expand_as(norm) # [batch, x, y, z]
        newnorm[newnorm < 0] = 0
        ratio = newnorm / norm
        ratio = ratio.view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x)

        self.save_for_backward(x, b)
        r = x * ratio
        time = time_logging.end("norm relu (forward)", time)
        return r

    def backward(self, grad_out): # pylint: disable=W
        time = time_logging.start()
        x, b = self.saved_tensors

        norm = torch.sqrt(torch.sum(x * x, dim=1)) + 1e-8 # [batch, x, y, z]
        newnorm = norm - b.expand_as(norm) # [batch, x, y, z]
        newnorm[newnorm < 0] = 0
        ratio = newnorm / norm
        ratio = ratio.view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x)

        grad_x = grad_out * ratio
        grad_x += torch.sum(grad_out * x, dim=1, keepdim=True).expand_as(x) * x / (norm ** 2).view(x.size(0), 1, x.size(2), x.size(3), x.size(4)).expand_as(x) * (1 - ratio)
        grad_x[ratio <= 0] = 0

        grad_b = -torch.sum(grad_out * x, dim=1) / norm
        grad_b[norm < b] = 0
        grad_b = torch.sum(grad_b.view(-1), dim=0)
        time = time_logging.end("norm relu (backward)", time)
        return grad_x, grad_b


def test_norm_relu_gradient(x=None, b=None):
    from se3_cnn.utils.test import gradient_approximation

    if x is None:
        x = torch.autograd.Variable(torch.rand(2, 5, 10, 10, 10), requires_grad=True)
    if b is None:
        b = torch.autograd.Variable(torch.rand(1), requires_grad=True)

    # Autograd
    y = NormReluFunction()(x, b)
    grad_y = torch.rand(2, 5, 10, 10, 10)
    torch.autograd.backward(y, grad_y)

    # Approximation
    grad_x_naive = gradient_approximation(lambda x: NormReluFunction().forward(x, b.data), x.data, grad_y, 1e-5)
    grad_b_naive = gradient_approximation(lambda b: NormReluFunction().forward(x.data, b), b.data, grad_y, 1e-5)

    print("grad_x {}".format(x.grad.data.std()))
    print("grad_x_naive {}".format(grad_x_naive.std()))
    print("grad_x - grad_x_naive {}".format((x.grad.data - grad_x_naive).std()))

    print("grad_b {}".format(b.grad.data))
    print("grad_b_naive {}".format(grad_b_naive))
    print("grad_b - grad_b_naive {}".format((b.grad.data - grad_b_naive)))

    return x.grad.data, grad_x_naive
