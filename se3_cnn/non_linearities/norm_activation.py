# pylint: disable=C,R,E1101
import torch


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
