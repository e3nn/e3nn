#pylint: disable=C,R,E1101
import torch

class SE3Dropout(torch.nn.Module):
    def __init__(self, Rs, p=0.5):
        super().__init__()
        self.Rs = Rs
        self.p = p

    def forward(self, x): # pylint: disable=W
        return SE3DropoutF(self.Rs, self.p if self.training else 0)(x)

class SE3DropoutF(torch.autograd.Function):
    def __init__(self, Rs, p=0.5):
        '''
        :param Rs: list of couple (multiplicity, dimension)
        '''
        super().__init__()
        self.p = p
        self.Rs = Rs
        self.noise = None

    def compute_noise(self, size):
        noises = []
        for mul, dim in self.Rs:
            noise = torch.FloatTensor(size[0], mul, *size[2:])

            if self.p == 1: noise.fill_(0)
            elif self.p == 0: noise.fill_(1)
            else: noise.bernoulli_(1 - self.p).div_(1 - self.p)

            noises.append(noise.repeat(1, dim, *(1,) * (len(size) - 2)))
        self.noise = torch.cat(noises, dim=1)

    def forward(self, x): # pylint: disable=W
        if self.noise is None:
            self.compute_noise(x.size())
        if x.is_cuda and not self.noise.is_cuda:
            self.noise = self.noise.cuda()
        return x * self.noise

    def backward(self, grad_y): #pylint: disable=W
        if grad_y.is_cuda and not self.noise.is_cuda:
            self.noise = self.noise.cuda()
        return grad_y * self.noise



def test_dropout_gradient():
    from se3_cnn.utils.test import gradient_approximation
    do = SE3DropoutF([(2, 1), (1, 3), (1, 5)], p=0.5)

    x = torch.autograd.Variable(torch.rand(2, 2 + 3 + 5, 10, 10, 10), requires_grad=True)

    y = do(x)
    grad_y = torch.rand(*y.size())
    torch.autograd.backward(y, grad_y)

    grad_x_approx = gradient_approximation(lambda x: do(torch.autograd.Variable(x)).data, x.data, grad_y, epsilon=1e-5)

    print("grad_x {}".format(x.grad.data.std()))
    print("grad_x_approx {}".format(grad_x_approx.std()))
    print("grad_x - grad_x_approx {}".format((x.grad.data - grad_x_approx).std()))

    return x.grad.data, grad_x_approx
