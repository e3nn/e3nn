# pylint: disable=C,R,E1101
import torch
import torch.nn.functional as F


def low_pass_filter(image, scale, stride=1):
    """
    :param tensor image: [..., x, y, z]
    :param float scale:
    :param int stride:
    """
    if scale <= 1:
        assert stride == 1
        return image

    sigma = 0.5 * (scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=image.dtype, device=image.device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(size, 1, 1).expand(size, size, size)
    y = rng.view(1, size, 1).expand(size, size, size)
    z = rng.view(1, 1, size).expand(size, size, size)

    kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size, size)

    out = F.conv3d(image.view(-1, 1, *image.size()[-3:]), kernel, padding=size // 2, stride=stride)
    out = out.view(*image.size()[:-3], *out.size()[-3:])
    return out
