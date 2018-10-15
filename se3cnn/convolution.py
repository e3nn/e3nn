# pylint: disable=C,R,E1101,E1102
import torch
from se3cnn import SE3Kernel
from se3cnn import kernel


class SE3Convolution(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, size, radial_window=kernel.gaussian_window_wrapper, dyn_iso=False, verbose=False, **kwargs):
        super().__init__()

        self.kernel = SE3Kernel(Rs_in, Rs_out, size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)
        self.kwargs = kwargs

    def __repr__(self):
        return "{name} ({kernel}, kwargs={kwargs})".format(
            name=self.__class__.__name__,
            kernel=self.kernel,
            kwargs=self.kwargs,
        )

    def forward(self, input):  # pylint: disable=W
        return torch.nn.functional.conv3d(input, self.kernel(), **self.kwargs)


def test_equivariance():
    torch.set_default_dtype(torch.float64)

    f = torch.nn.Sequential(
        SE3Convolution([(1, 0)], [(2, 0), (2, 1), (1, 2)], size=5, dyn_iso=True),
        SE3Convolution([(2, 0), (2, 1), (1, 2)], [(1, 0)], size=5, dyn_iso=False),
    )

    def rotate(t):
        # rotate 90 degrees in plane of axes 2 and 3
        return torch.flip(t, (2, )).transpose(2, 3)

    def unrotate(t):
        # undo the rotation by 3 more rotations
        return rotate(rotate(rotate(t)))

    inp = torch.randn(2, 1, 16, 16, 16)
    inp_r = rotate(inp)

    diff_inp = (inp - unrotate(inp_r)).abs().max().item()
    if diff_inp > 1e-10:
        raise AssertionError('Sanity check failed, `unrotate` doesn\'t work')

    out = f(inp)
    out_r = f(inp_r)

    diff_out = (out - unrotate(out_r)).abs().max().item()
    print("diff_out = {}".format(diff_out))
    if diff_out > 1e-10:
        raise AssertionError('Operations are not rotation-equivariant. Diff', diff_out)


def test_normalization(batch, input_size, Rs_in, Rs_out, size, dyn_iso):
    conv = SE3Convolution(Rs_in, Rs_out, size, dyn_iso=dyn_iso)

    print("Weights Number = {} Mean = {:.3f} Std = {:.3f}".format(conv.kernel.weight.numel(), conv.kernel.weight.mean().item(), conv.kernel.weight.std().item()))

    n_out = sum([m * (2 * l + 1) for m, l in Rs_out])
    n_in = sum([m * (2 * l + 1) for m, l in Rs_in])

    x = torch.randn(batch, n_in, input_size, input_size, input_size)
    print("x Number = {} Mean = {:.3f} Std = {:.3f}".format(x.numel(), x.mean().item(), x.std().item()))
    y = conv(x)

    assert y.size(1) == n_out

    print("y Number = {} Mean = {:.3f} Std = {:.3f}".format(y.numel(), y.mean().item(), y.std().item()))
    return y


def test_combination_gradient(Rs_in, Rs_out, size):
    conv = SE3Convolution(Rs_in, Rs_out, size).type(torch.float64)

    x = torch.rand(1, sum(m * (2 * l + 1) for m, l in Rs_in), 6, 6, 6, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(conv, (x, ), eps=1)


def main():
    test_equivariance()
    for dyn_iso in [False, True]:
        test_normalization(3, 15,
                           [(2, 0), (1, 1), (3, 4)],
                           [(2, 0), (2, 1), (1, 2)],
                           5, dyn_iso)
    test_combination_gradient([(1, 0), (1, 1)], [(1, 0)], 5)


if __name__ == "__main__":
    main()
