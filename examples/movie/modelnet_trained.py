# pylint: disable=C,R,E1101,E1102,W0221
import argparse
import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock
from se3cnn.util.dataset.modelnet10 import ModelNet10, Obj2Voxel, CacheNPY
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from se3cnn.util import time_logging
from se3cnn.SE3 import rotate_scalar
from se3cnn.SO3 import rot
from se3cnn.SO3 import irr_repr
import os
from threading import Thread


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (1, ),
            (4, 2, 1),
            (4, 2, 1),
            (2, )
        ]

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': None},
        ]

        common_block_params = {
            'size': 7,
            'stride': 1,
            'padding': 9,
            'dilation': 3,
            'normalization': 'batch',
        }

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(*blocks)
        self.post_activations = None

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        self.post_activations = []
        for op in self.sequence:
            x = op(x)
            self.post_activations.append(x)
        return x


def train(device, file):
    # classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    t = time_logging.start()

    def target_transform(x):
        if x == "chair":
            return 0
        else:
            return 1

    cache = CacheNPY("v128d", transform=Obj2Voxel(128, double=True))
    def transform(x):
        x = cache(x)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0) / 8

    dataset = ModelNet10(
        "./modelnet10/",
        "train",
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    dataset.files = [x for x in dataset.files if any(cl in x for cl in ["chair", "table", "desk", "dresser"])]

    model = Model().to(device)
    if os.path.exists(file):
        print("load model")
        model.load_state_dict(torch.load(file))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)

    t = time_logging.end("init", t)

    for _ in range(10):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            t = time_logging.end("load", t)
            output = model(x)
            del x

            output = output.view(output.size(0), output.size(1), -1).mean(-1)
            loss = F.cross_entropy(output, y)
            print("{}/{} loss={:.3g}  {}".format(
                i,
                len(dataloader),
                loss.item(),
                ", ".join("{}:{}".format(a, b) for a, b in zip(output.argmax(1), y))
            ))
            t = time_logging.end("->", t)

            del output, y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t = time_logging.end("<-", t)
            del loss

            if i % 10 == 0:
                print("save...")
                def noInterrupt():
                    torch.save(model.state_dict(), file)

                a = Thread(target=noInterrupt)
                a.start()
                a.join()
                t = time_logging.end("save", t)
                print(time_logging.text_statistics())



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
    if x.ndimension() == 4 and x.size(0) == 3:
        rep = irr_repr(1, alpha, beta, gamma, x.dtype).to(x.device)
        x = torch.einsum("ij,jxyz->ixyz", (rep, x))
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


def record(device, pickle_file, movie_file, n_frames):
    t = time_logging.start()

    cache = CacheNPY("v128d", transform=Obj2Voxel(128, double=True))
    def transform(x):
        x = cache(x)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0) / 8

    dataset = ModelNet10(
        "./modelnet10/",
        "train",
        download=True,
        transform=transform,
    )
    x, _ = dataset[628]
    x = F.pad(x.unsqueeze(0), (45,) * 6)[0, 0]
    x = x.to(device)

    model = Model().to(device)
    model.load_state_dict(torch.load(pickle_file))

    def f(x):
        t = time_logging.start()
        while x.ndimension() < 5:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = model(x)

        while x.size(0) == 1:
            x = x[0]
        time_logging.end("model", t)
        return model.post_activations[-2][0, 0], model.post_activations[-2][0, 4:7]

    alpha = np.linspace(0, 2 * np.pi, n_frames)
    beta = np.linspace(0, np.pi / 2, n_frames)

    fig = plt.figure(figsize=(4 * 6, 3 * 6))

    ax_input = [fig.add_axes([x, 2/3, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_input = [None] * 4
    ax_scalar = [fig.add_axes([x, 1/3, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_scalar = [None] * 4
    ax_vector = [fig.add_axes([x, 0, 1/4, 1/3]) for x in [0, 1/4, 1/2, 3/4]]
    im_vector = [None] * 4

    s, v = f(x)

    imshow_param = {
        'interpolation': 'none',
        'origin': 'lower',
    }

    quiver_param = {
        'units': 'xy',
        'scale': 0.5,
        'pivot': 'tail',
        'headwidth': 2.5,
        'headlength': 5,
        'minlength': 0,
        'minshaft': 1
    }

    print(s.mean().item(), s.std().item())

    for i in range(4):
        im_input[i] = ax_input[i].imshow(project(x, 0), **imshow_param, vmin=0, vmax=0.1)
        im_scalar[i] = ax_scalar[i].imshow(project(s, 0), **imshow_param, vmin=-3, vmax=7)
        im_vector[i] = ax_vector[i].quiver(*project_vector(v, 0), **quiver_param)

    for im in im_input + im_scalar + im_vector:
        im.set_cmap("summer")

    for ax in ax_input + ax_scalar + ax_vector:
        ax.set_axis_off()

    time_logging.end("init", t)

    def init():
        s, v = f(x)

        for i in range(4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim))
            im_scalar[i].set_data(project(s, dim))
            im_vector[i].set_UVC(*project_vector(v, dim))

        return tuple(im_input + im_scalar + im_vector)

    def animate(i):
        print("rendering {} / {}".format(i, n_frames))
        a, b = alpha[i], beta[i]

        rx = rotate(x, a, b, 0)
        s, v = f(rx)
        rs = rotate(s, 0, -b, -a)
        rv = rotate(v, 0, -b, -a)

        for i in range(0, 2):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(rx, dim))
            im_scalar[i].set_data(project(s, dim))
            im_vector[i].set_UVC(*project_vector(v, dim))

        for i in range(2, 4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim))
            im_scalar[i].set_data(project(rs, dim))
            im_vector[i].set_UVC(*project_vector(rv, dim))

        return tuple(im_input + im_scalar + im_vector)

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=n_frames)

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=12, bitrate=3000)
    ani.save(movie_file, writer=writer)

    print(time_logging.text_statistics())


def main():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--movie", type=str)
    parser.add_argument("--n_frames", type=int, default=181)
    parser.add_argument("--action", choices={"train", "record"}, required=True)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')

    if args.action == "train":
        train(device, args.pickle)

    if args.action == "record":
        assert args.movie
        record(device, args.pickle, args.movie, args.n_frames)


main()
