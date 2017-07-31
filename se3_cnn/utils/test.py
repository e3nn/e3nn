"""
Useful functions to make tests
"""
import torch


def gradient_approximation(function, inpu, grad_out, epsilon):
    """
    Computes the gradient with finite difference
    """
    grad_x = torch.zeros(inpu.numel())

    for i in range(inpu.numel()):
        inpu_ = inpu.clone().view(-1)
        inpu_[i] += epsilon
        out_plus = function(inpu_.view(*inpu.size()))

        inpu_ = inpu.clone().view(-1)
        inpu_[i] -= epsilon
        out_minus = function(inpu_.view(*inpu.size()))

        diff = (out_plus - out_minus) / (2 * epsilon)
        grad_x[i] = torch.sum(grad_out * diff)

    return grad_x.view(*inpu.size())
