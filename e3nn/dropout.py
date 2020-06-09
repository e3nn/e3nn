# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name
import torch
from e3nn import rs


class Dropout(torch.nn.Module):
    def __init__(self, Rs, p=0.5):
        super().__init__()
        self.Rs = rs.convention(Rs)
        self.p = p

    def __repr__(self):
        return "{} (p={})".format(self.__class__.__name__, self.p)

    def forward(self, x):
        """
        :param x: tensor [batch, ..., channel]
        """
        if not self.training:
            return x

        noises = []
        for mul, l, _p in self.Rs:
            dim = 2 * l + 1
            noise = x.new_empty(x.shape[0], mul)

            if self.p == 1:
                noise.fill_(0)
            elif self.p == 0:
                noise.fill_(1)
            else:
                noise.bernoulli_(1 - self.p).div_(1 - self.p)

            noise = noise.unsqueeze(2).expand(-1, -1, dim).reshape(x.shape[0], mul * dim)
            noises.append(noise)

        noise = torch.cat(noises, dim=-1)
        return x * noise.reshape(x.shape[0], *(1,) * (x.dim() - 2), x.shape[-1])
