# pylint: disable=C,R,E1101,E1102
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
import os


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        features = [
            (1,),
            (2, 2, 2, 2),
            (2, 2, 2, 2),
            (2, 2, 2, 2),
            (1,),
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


def rotate(x, alpha):
    t = time_logging.start()
    y = x.cpu().detach().numpy()
    R = rot(alpha, 0, 0)
    y = rotate_scalar(y, R)
    x = x.new_tensor(y)
    time_logging.end("rotate", t)
    return x


def project(x):
    return x.mean(2).detach().cpu().numpy()


def main():
    device = torch.device("cuda:0")
    f = Model().to(device)

    x = np.load("{}/119.npy".format(os.path.dirname(__file__)))
    x = np.pad(x, 30, "constant")
    x = x.astype(np.float32)
    x = torch.tensor(x, device=device, dtype=torch.float32)

    n = 61
    angles = np.linspace(0, np.pi, n)

    fig = plt.figure(figsize=(13, 13))

    ax1 = fig.add_axes([0, 0, .5, .5])  # x
    ax2 = fig.add_axes([0, .5, .5, .5])  # R(x)
    ax3 = fig.add_axes([.5, 0, .5, .5])  # f(R(x))
    ax4 = fig.add_axes([.5, .5, .5, .5])  # R-1(f(R(x)))

    y = f(x)
    image1 = ax1.imshow(project(x), interpolation='none')
    image2 = ax2.imshow(project(x), interpolation='none')
    image3 = ax3.imshow(project(y), interpolation='none')
    image4 = ax4.imshow(project(y), interpolation='none')

    for image in [image1, image2, image3, image4]:
        image.set_cmap("summer")

    text = ax1.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='white', fontsize=40)

    for ax in [ax1, ax2, ax3, ax4]:
        plt.sca(ax)
        plt.axis('off')

    def init():
        image2.set_data(project(x))
        y = f(x)
        image3.set_data(project(y))
        image4.set_data(project(y))
        text.set_text("")
        return image2, image3, image4, text

    def animate(i):
        print(i)
        alpha = angles[i]

        rx = rotate(x, alpha)
        frx = f(rx)
        rfrx = rotate(frx, -alpha)
        image2.set_data(project(rx))
        image3.set_data(project(frx))
        image4.set_data(project(rfrx))
        text.set_text(r"${} \pi / {}$".format(i, n - 1))
        return image2, image3, image4, text

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=n)

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("movie.mp4", writer=writer)

    print(time_logging.text_statistics())


main()
