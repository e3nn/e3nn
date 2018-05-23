# pylint: disable=C,R,E1101
import torch


class SE3Dropout(torch.nn.Module):
    def __init__(self, Rs, p=0.5):
        super().__init__()
        self.Rs = [(mul, dim) for mul, dim in Rs if mul * dim > 0]
        self.p = p

    def __repr__(self):
        return "{} (p={})".format(
            self.__class__.__name__,
            self.p)

    def forward(self, x):  # pylint: disable=W
        p = self.p if self.training else 0

        noises = []
        for mul, dim in self.Rs:
            noise = x.new_empty(x.size(0), mul, 1, 1, 1)  # independent of spatial position

            if p == 1:
                noise.fill_(0)
            elif p == 0:
                noise.fill_(1)
            else:
                noise.bernoulli_(1 - p).div_(1 - p)

            noise = noise.unsqueeze(2).expand(-1, -1, dim, -1, -1, -1).contiguous().view(x.size(0), mul * dim, 1, 1, 1)
            noises.append(noise)
        noise = torch.cat(noises, dim=1)
        return x * noise
