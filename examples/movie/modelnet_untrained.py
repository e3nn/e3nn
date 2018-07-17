# pylint: disable=C,R,E1101,E1102,W0221
import torch
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock
from se3cnn.SE3 import rotate_scalar
from se3cnn.SO3 import rot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from se3cnn.util import time_logging
from se3cnn.SO3 import irr_repr
from se3cnn.util.dataset.modelnet10 import ModelNet10, Obj2Voxel, CacheNPY


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        features = [
            (1,),
            (2, 2, 2, 2),
            (0, 2, 2, 2),  # no scalar fields here
            (2, 2, 2, 2),
            (1, 1),
        ]

        common_block_params = {
            'size': 5,
            'padding': 2,
        }

        block_params = [
            { 'activation': (F.relu, F.sigmoid) },
            { 'activation': (F.relu, F.sigmoid) },
            { 'activation': (F.relu, F.sigmoid) },
            { 'activation': None },
        ]

        assert len(block_params) + 1 == len(features)

        self.blocks = nn.Sequential(*(GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))))

    def forward(self, x):
        t = time_logging.start()
        while x.ndimension() < 5:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = self.blocks(x)

        while x.size(0) == 1:
            x = x[0]
        time_logging.end("model", t)
        return x


def rotate(x, alpha, beta, gamma):
    t = time_logging.start()
    y = x.cpu().detach().numpy()
    R = rot(alpha, beta, gamma)
    if x.ndimension() == 4:
        for i in range(y.shape[0]):
            y[i] = rotate_scalar(y[i], R)
    else:
        y = rotate_scalar(y, R)
    x = x.new_tensor(y)
    rep = irr_repr(1, alpha, beta, gamma, x.dtype).to(x.device)
    if x.ndimension() == 4 and x.size(0) == 1 + 3:
        x[1:] = torch.einsum("ij,jxyz->ixyz", (rep, x[1:]))
    time_logging.end("rotate", t)
    return x


def project(x, dim):
    n = x.size(0)
    assert x.size(1) == n
    assert x.size(2) == n
    n = int(0.15 * n)
    x = x.mean(dim)
    x = x[n:-n, n:-n]
    if dim == 0:
        x = x.t().contiguous()
    return x.detach().cpu().numpy()


def project_vector(x, dim):
    A = x.new_tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).t()
    x = torch.einsum("ij,jxyz->ixyz", (A, x))
    if dim == 0:
        return project(x[2], 0), project(x[1], 0)
    if dim == 1:
        return project(x[0], 1), project(x[2], 1)
    if dim == 2:
        return project(x[0], 2), project(x[1], 2)


def main():
    t = time_logging.start()
    device = torch.device("cuda:0")

    torch.manual_seed(1)
    f = Model().to(device)


    cache = CacheNPY("v100d", transform=Obj2Voxel(100, double=True))

    def transform(x):
        x = cache(x)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0) / 8

    dataset = ModelNet10(
        "./modelnet10/",
        "train",
        download=True,
        transform=transform,
    )

    x, _ = dataset[119]
    x = F.pad(x.unsqueeze(0), (30,) * 6)[0, 0]
    x = x.to(device)

    n = 181
    # x = x[::2, ::2, ::2]
    # n = 4

    alpha = np.linspace(0, 2 * np.pi, n)
    beta = np.linspace(0, np.pi / 2, n)

    fig = plt.figure(figsize=(4 * 6, 3 * 6))

    ax_input = [fig.add_axes([x, 2/3, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_input = [None] * 4
    ax_scalar = [fig.add_axes([x, 1/3, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_scalar = [None] * 4
    ax_vector = [fig.add_axes([x, 0, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_vector = [None] * 4

    y = f(x)

    imshow_param = {
        'interpolation': 'none',
        'origin': 'lower',
    }

    quiver_param = {
        'units': 'xy',
        'scale': 0.2,
        'pivot': 'tail',
        'headwidth': 2.5,
        'headlength': 5,
        'minlength': 0,
        'minshaft': 1
    }

    for i in range(4):
        im_input[i] = ax_input[i].imshow(project(x, 0), **imshow_param)
        im_scalar[i] = ax_scalar[i].imshow(project(y[0], 0), **imshow_param)
        im_vector[i] = ax_vector[i].quiver(*project_vector(y[1:], 0), **quiver_param)

    for im in im_input + im_scalar + im_vector:
        im.set_cmap("summer")

    # ax1.text(0.5, 0.99, 'stabilized input', horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes, color='white', fontsize=30)
    # ax2.text(0.5, 0.99, 'input', horizontalalignment='center', verticalalignment='top', transform=ax2.transAxes, color='white', fontsize=30)
    # ax3.text(0.5, 0.99, 'featuremap', horizontalalignment='center', verticalalignment='top', transform=ax3.transAxes, color='white', fontsize=30)
    # ax4.text(0.5, 0.99, 'stabilized featuremap', horizontalalignment='center', verticalalignment='top', transform=ax4.transAxes, color='white', fontsize=30)

    for ax in ax_input + ax_scalar + ax_vector:
        ax.set_axis_off()

    time_logging.end("init", t)

    def init():
        y = f(x)

        for i in range(4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim))
            im_scalar[i].set_data(project(y[0], dim))
            im_vector[i].set_UVC(*project_vector(y[1:], dim))

        return tuple(im_input + im_scalar + im_vector)

    def animate(i):
        print("rendering {} / {}".format(i, n))
        a, b = alpha[i], beta[i]

        rx = rotate(x, a, b, 0)
        frx = f(rx)
        rfrx = rotate(frx, 0, -b, -a)

        for i in range(0, 2):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(rx, dim))
            im_scalar[i].set_data(project(frx[0], dim))
            im_vector[i].set_UVC(*project_vector(frx[1:], dim))

        for i in range(2, 4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim))
            im_scalar[i].set_data(project(rfrx[0], dim))
            im_vector[i].set_UVC(*project_vector(rfrx[1:], dim))

        return tuple(im_input + im_scalar + im_vector)

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=n)

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=12, bitrate=3000)
    ani.save("movie.mp4", writer=writer)

    print(time_logging.text_statistics())


main()
