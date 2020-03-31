# pylint: disable=no-member, missing-docstring, invalid-name, redefined-builtin, arguments-differ, line-too-long
import torch

from e3nn.non_linearities import GatedBlock, ScalarActivation
from e3nn.non_linearities.rescaled_act import tanh


class GRU(torch.nn.Module):
    """
    Simplified version of GRU

    - there is no input x
    - there is no gate r (remember gate)
    """
    def __init__(self, repr, Operation):
        """
        :param repr: multiplicities
        :param Operation: class of signature (Rs_out)
        """
        super().__init__()

        self.repr = repr

        self.z_conv = Operation([(mul, l) for l, mul in enumerate(repr)], [(sum(repr), 0)])
        self.z_act = ScalarActivation([(sum(repr), torch.sigmoid)], bias=False)

        gb = GatedBlock(repr, tanh, tanh)
        self.h_tilde_op = Operation(gb.Rs_in)
        self.h_tilde_gb = gb

    def forward(self, h, *args, **kwargs):
        """
        :param h: tensor [batch, channel, ...]
        :return: tensor [batch, channel, ...]
        """
        batch, _, *size = h.size()

        z = self.z_conv(h, *args, **kwargs)  # [batch, channel, ...]
        z = self.z_act(z)  # [batch, channel, ...]

        h_tilde = self.h_tilde_op(h, *args, **kwargs)
        h_tilde = self.h_tilde_gb(h_tilde)

        outs = []
        i = 0
        j = 0
        for l, mul in enumerate(self.repr):
            d = mul * (2 * l + 1)
            h_ = h[:, i: i + d].reshape(batch, mul, 2 * l + 1, *size)
            h_tilde_ = h_tilde[:, i: i + d].reshape(batch, mul, 2 * l + 1, *size)
            z_ = z[:, j: j + mul].view(batch, mul, 1, *size)
            i += d
            j += mul

            out = (1 - z_) * h_ + z_ * h_tilde_
            outs.append(out.view(batch, -1, *size))

        return torch.cat(outs, dim=1)
