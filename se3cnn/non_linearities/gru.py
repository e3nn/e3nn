# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch

from se3cnn.non_linearities import GatedBlock, ScalarActivation
from se3cnn.non_linearities.rescaled_act import sigmoid, tanh


class GRU(torch.nn.Module):
    """
    Simplified version of GRU

    - there is no input x
    - there is no gate r (remember gate)
    """
    def __init__(self, repr, Convolution):
        """
        :param repr: multiplicities
        :param Convolution: class of signature (Rs_in, Rs_out)
        """
        super().__init__()

        self.repr = repr

        self.z_conv = Convolution([(mul, l) for l, mul in enumerate(repr)], [(sum(repr), 0)])
        self.z_act = ScalarActivation([(sum(repr), sigmoid)], bias=False)

        self.h_tilde = GatedBlock(repr, repr, tanh, tanh, Convolution)


    def forward(self, h, *args, **kwargs):
        """
        :param h: tensor [batch, channel, ...]
        :return: tensor [batch, channel, ...]
        """
        batch, _, *size = h.size()

        z = self.z_conv(h, *args, **kwargs)  # [batch, channel, ...]
        z = self.z_act(z)  # [batch, channel, ...]

        h_tilde = self.h_tilde(h, *args, **kwargs)

        outs = []
        i = 0
        j = 0
        for l, mul in enumerate(self.repr):
            d = mul * (2 * l + 1)
            h_ = h[:, i: i+d].contiguous().view(batch, mul, 2 * l + 1, *size)
            h_tilde_ = h_tilde[:, i: i+d].contiguous().view(batch, mul, 2 * l + 1, *size)
            z_ = z[:, j: j+mul].view(batch, mul, 1, *size)
            i += d
            j += mul

            out = (1 - z_) * h_ + z * h_tilde_
            outs.append(out.view(batch, -1, *size))

        return torch.cat(outs, dim=1)
