# pylint: disable=C,R,E1101,E1102,W0221
import argparse
import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock
from se3cnn.util.dataset.shapes import ModelNet10, Obj2Voxel, CacheNPY, EqSampler
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from se3cnn.util import time_logging
from se3cnn.SE3 import rotate_scalar
from se3cnn.SO3 import rot
from se3cnn.SO3 import irr_repr
import os
from threading import Thread


def low_pass_filter(image, scale):
    """
    :param image: [..., x, y, z]
    :param scale: float
    """
    if scale >= 1:
        return image

    dtype = image.dtype
    device = image.device

    sigma = 0.5 * (1 / scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=dtype, device=device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(size, 1, 1).expand(size, size, size)
    y = rng.view(1, size, 1).expand(size, size, size)
    z = rng.view(1, 1, size).expand(size, size, size)

    kernel = torch.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    out = F.conv3d(image.view(-1, 1, *image.size()[-3:]), kernel.view(1, 1, size, size, size), padding=size//2)
    out = out.view(*image.size())
    return out


class SE3Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (1, ),
            (4, 2, 1),
            (4, 2, 1),
            (2, )
        ]

        block_params = [
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': None},
        ]

        common_block_params = {
            'size': 7,
            'stride': 1,
            'padding': 9,
            'dilation': 3,
            'normalization': None,
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
        x = low_pass_filter(x, 1 / 3)  # dilation == 3

        self.post_activations = []
        for op in self.sequence:
            x = op(x)
            self.post_activations.append(x)
        return x


class BaselineModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            1,
            15,
            15,
            2
        ]

        common_block_params = {
            'kernel_size': 7,
            'stride': 1,
            'padding': 9,
            'dilation': 3,
        }

        blocks = []
        for i in range(len(features) - 1):
            blocks += [
                nn.Conv3d(features[i], features[i + 1], **common_block_params),
                nn.ReLU(inplace=True)
            ]

        self.sequence = torch.nn.Sequential(*blocks)
        self.post_activations = None

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = low_pass_filter(x, 1 / 3)  # dilation == 3

        self.post_activations = []
        for op in self.sequence:
            x = op(x)
            self.post_activations.append(x)
        return x


def train(device, file, modelname, batch_size, rotate):
    # classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    t = time_logging.start()

    def target_transform(x):
        if x == "chair":
            return 0
        else:
            return 1

    cache = CacheNPY("v128d" if rotate == 0 else "v128dr", transform=Obj2Voxel(128, double=True, rotate=(rotate > 0)), repeat=rotate)
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

    if modelname == "se3":
        model = SE3Model().to(device)
    if modelname == "baseline":
        model = BaselineModel().to(device)

    if os.path.exists(file):
        print("load model")
        model.load_state_dict(torch.load(file))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=2, sampler=EqSampler(dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)

    t = time_logging.end("init", t)

    for _ in range(10):
        correct = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            t = time_logging.end("load", t)
            output = model(x)
            model.post_activations = None
            del x

            output = output.view(output.size(0), output.size(1), -1).mean(-1)
            loss = F.cross_entropy(output, y)
            correct += (output.argmax(1) == y).long().sum().item()

            print("{}/{} loss={:.3g}  {}  {:.1f}%".format(
                i,
                len(dataloader),
                loss.item(),
                ", ".join("{}:{}".format(a, b) for a, b in zip(output.argmax(1), y)),
                100 * correct / ((i + 1) * output.size(0)),
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


def project(x, dim, crop=0):
    n = x.size(0)
    assert x.size(1) == n
    assert x.size(2) == n
    n = int(crop * n)
    x = x[n:-n, n:-n, n:-n] if n > 0 else x
    x = x.mean(dim)
    if dim == 0:
        x = x.t().contiguous()
    return x.detach().cpu().numpy()


def project_vector(x, dim, crop=0):
    A = x.new_tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).t()
    x = torch.einsum("ij,jxyz->ixyz", (A, x))
    if dim == 0:
        u, v = project(x[1], 0, crop), project(x[2], 0, crop)
    if dim == 1:
        u, v = project(x[2], 1, crop), project(x[0], 1, crop)
    if dim == 2:
        u, v = project(x[1], 2, crop), project(x[0], 2, crop)

    n = u.shape[0]
    np.random.seed(15439338)
    x, y = np.random.rand(2, n, n) * (n - 1)

    ix = np.floor(x).astype(np.int)
    dx = x - ix

    iy = np.floor(y).astype(np.int)
    dy = y - iy

    nu = (1-dx)*(1-dy) * u[ix, iy] + (dx)*(1-dy) * u[ix+1, iy] + (dx)*(dy) * u[ix+1, iy+1] + (1-dx)*(dy) * u[ix, iy+1]
    nv = (1-dx)*(1-dy) * v[ix, iy] + (dx)*(1-dy) * v[ix+1, iy] + (dx)*(dy) * v[ix+1, iy+1] + (1-dx)*(dy) * v[ix, iy+1]

    return y, x, nu, nv


def record(device, pickle_file, movie_file, n_frames, objid, modelname):
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
    x, _ = dataset[objid]
    x = F.pad(x.unsqueeze(0), (45,) * 6)[0, 0]
    x = x.to(device)

    if modelname == "se3":
        model = SE3Model().to(device)
    if modelname == "baseline":
        model = BaselineModel().to(device)

    model.load_state_dict(torch.load(pickle_file))
    model.eval()

    # from IPython import embed; embed()

    def f(x):
        t = time_logging.start()
        while x.ndimension() < 5:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = model(x)

        time_logging.end("model", t)
        return model.post_activations[-2][0, 3], model.post_activations[-2][0, 4:7]

    alpha = np.concatenate((np.linspace(0, 2 * np.pi, n_frames - n_frames // 50), np.zeros((n_frames // 50,))))
    beta = np.concatenate((np.zeros((n_frames // 3,)), np.linspace(0, 2 * np.pi, n_frames - n_frames // 3 - n_frames // 50), np.zeros((n_frames // 50,))))

    a, topmargin, leftmargin = 6, 0.7, 0.7
    width = 4 * a + leftmargin
    height = 3 * a + topmargin
    fig = plt.figure(figsize=(width, height))

    x_a = a / width
    y_a = a / height
    x_m = leftmargin / width
    y_m = topmargin / height

    ax_input = [fig.add_axes([x_m + i * x_a, 2 * y_a, x_a, y_a]) for i in range(4)]
    im_input = [None] * 4
    ax_scalar = [fig.add_axes([x_m + i * x_a, y_a, x_a, y_a]) for i in range(4)]
    im_scalar = [None] * 4
    ax_vector = [fig.add_axes([x_m + i * x_a, 0, x_a, y_a]) for i in range(4)]
    im_vector = [None] * 4

    color, fontsize = 'black', 25
    fig.text(x_m / 2, 2.5 * y_a, 'input (scalar field)', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=90)
    fig.text(x_m / 2, 1.5 * y_a, 'output (scalar field)', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=90)
    fig.text(x_m / 2, 0.5 * y_a, 'output (vector field)', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=90)

    fig.text(x_m + 0.5 * x_a, 3 * y_a + 0.5 * y_m, 'top view', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=0)
    fig.text(x_m + 1.5 * x_a, 3 * y_a + 0.5 * y_m, 'side view', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=0)
    fig.text(x_m + 2.5 * x_a, 3 * y_a + 0.5 * y_m, 'top view stabilized', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=0)
    fig.text(x_m + 3.5 * x_a, 3 * y_a + 0.5 * y_m, 'side view stabilized', horizontalalignment='center', verticalalignment='center', color=color, fontsize=fontsize, rotation=0)

    s, v = f(x)
    print("positive=table, negative=chair : {:.3g}".format(s.mean().item()))

    imshow_param = {
        'interpolation': 'none',
        'origin': 'lower',
    }

    quiver_param = {
        'units': 'xy',
        'scale': 2,
        'pivot': 'tail',
        'headwidth': 2.5,
        'headlength': 5,
        'minlength': 0,
        'minshaft': 1
    }

    s_crop, so_crop, v_crop = 0.15, 0.08, 0.35
    bg = project(s, 0, so_crop)[0, 0]
    st = s.std().item()
    print(bg, st)

    for i in range(4):
        im_input[i] = ax_input[i].imshow(project(x, 0, s_crop), **imshow_param, vmin=0, vmax=0.1, cmap='gray')
        im_scalar[i] = ax_scalar[i].imshow(project(s, 0, so_crop), **imshow_param, vmin=bg-3*st, vmax=bg+3*st, cmap='viridis')
        im_vector[i] = ax_vector[i].quiver(*project_vector(v, 0, v_crop), **quiver_param)

    # from IPython import embed; embed()

    for ax in ax_input + ax_scalar + ax_vector:
        ax.set_axis_off()

    time_logging.end("init", t)

    def init():
        s, v = f(x)

        for i in range(4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim, s_crop))
            im_scalar[i].set_data(project(s, dim, so_crop))
            im_vector[i].set_UVC(*project_vector(v, dim, v_crop)[2:])

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
            im_input[i].set_data(project(rx, dim, s_crop))
            im_scalar[i].set_data(project(s, dim, so_crop))
            im_vector[i].set_UVC(*project_vector(v, dim, v_crop)[2:])

        for i in range(2, 4):
            dim = 2 if i % 2 == 0 else 0
            im_input[i].set_data(project(x, dim, s_crop))
            im_scalar[i].set_data(project(rs, dim, so_crop))
            im_vector[i].set_UVC(*project_vector(rv, dim, v_crop)[2:])

        return tuple(im_input + im_scalar + im_vector)

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=n_frames)

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=16, bitrate=3000)
    ani.save(movie_file, writer=writer)

    print(time_logging.text_statistics())


def main():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--movie", type=str)
    parser.add_argument("--objid", type=int, default=628)
    # office chair : 628
    # old fancy table: 3265
    # tri-table : 3644

    parser.add_argument("--n_frames", type=int, default=181)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--action", choices={"train", "record"}, required=True)
    parser.add_argument("--model", choices={"se3", "baseline"}, required=True)
    parser.add_argument('--rotate', type=int, default=0)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')

    if args.action == "train":
        train(device, args.pickle, args.model, args.batch_size, args.rotate)

    if args.action == "record":
        assert args.movie
        record(device, args.pickle, args.movie, args.n_frames, args.objid, args.model)


main()
