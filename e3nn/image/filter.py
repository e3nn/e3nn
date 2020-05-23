# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, line-too-long, no-member, invalid-name
import torch
import torch.nn.functional as F


class LowPassFilter(torch.nn.Module):
    def __init__(self, scale, stride=1):
        """
        :param float scale:
        :param int stride:
        """
        super().__init__()

        sigma = 0.5 * (scale ** 2 - 1) ** 0.5

        size = int(1 + 2 * 2.5 * sigma)
        if size % 2 == 0:
            size += 1

        rng = torch.arange(size, dtype=torch.get_default_dtype()) - size // 2  # [-(size // 2), ..., size // 2]
        x = rng.reshape(size, 1, 1).expand(size, size, size)
        y = rng.reshape(1, size, 1).expand(size, size, size)
        z = rng.reshape(1, 1, size).expand(size, size, size)

        kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, size, size, size)
        self.register_buffer('kernel', kernel)

        self.scale = scale
        self.stride = stride
        self.size = size

    def forward(self, image):
        """
        :param tensor image: [..., x, y, z, channel]
        """
        if self.scale <= 1:
            assert self.stride == 1
            return image

        out = image.reshape(-1, *image.shape[-4:])
        out = torch.einsum('txyzi->tixyz', out)
        out = out.reshape(-1, 1, *out.shape[-3:])
        out = F.conv3d(out, self.kernel, padding=self.size // 2, stride=self.stride)
        out = out.reshape(-1, image.shape[-1], *out.shape[-3:])
        out = torch.einsum('tixyz->txyzi', out)
        out = out.reshape(*image.shape[:-4], *out.shape[-4:])
        return out
