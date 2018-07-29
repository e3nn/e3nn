# pylint: disable=E1101,R,C,W1202,E1102
from se3cnn.util.dataset.shapes import ModelNet10
import math
import random
import subprocess
import numpy as np
import os

import torch
from se3cnn.SO3 import irr_repr, xyz_vector_basis_to_spherical_basis
import torch.nn.functional as F

from se3cnn.blocks import GatedBlock


class Obj2OrientedVoxel:
    def __init__(self, size, double=False):
        self.size = size
        self.double = double

    def __call__(self, file_path):
        tmpfile = '%030x.npy' % random.randrange(16**30)
        command = ["obj2voxel", "--size", str(self.size), file_path, tmpfile]

        c = 2 * math.pi * random.random()
        b = math.acos(random.random() * 2 - 1)
        a = 2 * math.pi * random.random()

        command += ["--alpha_rot", str(a)]
        command += ["--beta_rot", str(b)]
        command += ["--gamma_rot", str(c)]

        if self.double:
            command += ["--double"]

        subprocess.run(command)
        x = np.load(tmpfile).astype(np.int8).reshape((self.size, self.size, self.size))
        os.remove(tmpfile)
        return x, (a, b, c)


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (1, 2),
            (4, 4, 4),
            (8, 8, 4),
            (16, 16, 4),
            (32, 16, 4),
            (0, 2)
        ]

        common_block_params = {
            'size': 5,
            'stride': 2,
            'padding': 4,
            'normalization': None,
            'capsule_dropout_p': 0.0
        }

        block_params = [
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
        )

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        return self.sequence(x)


def rotation_from_orientations(t1, f1, t2, f2):
    from se3cnn.SO3 import x_to_alpha_beta, rot

    zero = t1.new_tensor(0)

    r_e_t1 = rot(*x_to_alpha_beta(t1), zero)
    r_e_t2 = rot(*x_to_alpha_beta(t2), zero)

    f1_e = r_e_t1.t() @ f1
    f2_e = r_e_t2.t() @ f2

    c = torch.atan2(f2_e[1], f2_e[0]) - torch.atan2(f1_e[1], f1_e[0])
    r_f1_f2 = rot(zero, zero, c)

    r = r_e_t2 @ r_f1_f2 @ r_e_t1.t()

    # t2 = r @ t1
    # f2 ~ r @ f1
    return r


def angle_from_rotation(r):
    return ((r.trace() - 1) / 2).acos()


def overlap(a, b):
    na = a.pow(2).sum(-1).pow(0.5)
    nb = b.pow(2).sum(-1).pow(0.5)
    return (a * b).sum(-1) / (na * nb + 1e-6)


def train_step(item, top, front, model, optimizer):
    from se3cnn.SO3 import rot

    model.train()

    abc = 15 * (torch.rand(3) * 2 - 1) / 180 * math.pi
    r = rot(*abc).to(item.device)
    tip = torch.cat([torch.einsum("ij,...j->...i", (r, x)) for x in [top, front]], dim=1)
    tip = tip.view(tip.size(0), tip.size(1), 1, 1, 1).expand(tip.size(0), tip.size(1), item.size(2), item.size(3), item.size(4))

    input = torch.cat([item, tip], dim=1)

    prediction = model(input)
    pred_top, pred_front = prediction[:, :3], prediction[:, 3:]

    overlap_top = overlap(pred_top, top)
    overlap_front = overlap(pred_front, front)
    overlap_orth = overlap(pred_top, pred_front)

    optimizer.zero_grad()
    (-overlap_top - overlap_front + overlap_orth.abs()).mean().backward()
    optimizer.step()

    a = torch.mean(torch.tensor([
        angle_from_rotation(rotation_from_orientations(top[i], front[i], pred_top[i], pred_front[i]))
        for i in range(len(item))
    ]))

    return overlap_top.mean().item(), overlap_front.mean().item(), overlap_orth.abs().mean().item(), a.item()


def main():
    def transform(x):
        convert = Obj2OrientedVoxel(32)
        x, abc = convert(x)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        rot = irr_repr(1, *abc) @ xyz_vector_basis_to_spherical_basis()
        top = rot @ torch.tensor([0, 0, 1.])
        front = rot @ torch.tensor([0, 1., 0])
        return x, (top, front)

    dataset = ModelNet10(
        "./modelnet10/",
        "train",
        download=True,
        transform=transform,
    )
    dataset.files = [x for x in dataset.files if "chair" in x]

    device = torch.device("cuda:0")
    model = Model().to(device)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        for (item, (top, front)), _ in train_loader:

            item, top, front = item.to(device), top.to(device), front.to(device)
            top, front, orth, angle = train_step(item, top, front, model, optimizer)

            print("{} avg_top={:.3f} avg_front={:.3f} avg_orth={:.3f} avg_angle={:.1f}".format(epoch, top, front, orth, 180 * angle / 3.14159))


if __name__ == "__main__":
    main()