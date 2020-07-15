# pylint: disable=invalid-name, no-member, arguments-differ, missing-docstring, line-too-long
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from e3nn.non_linearities import rescaled_act
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.radial import CosineBasisModel
from e3nn import o3

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GravityNet(torch.nn.Module):
    def __init__(self, num_radial=30, max_radius=2):
        super().__init__()

        sp = rescaled_act.Softplus(beta=5)
        RadialModel = partial(CosineBasisModel, max_radius=max_radius, number_of_basis=num_radial, h=100, L=2, act=sp)

        self.conv = Convolution(Kernel([(1, 0)], [(1, 1)], RadialModel))

    def forward(self, features, geometry):
        features = self.conv(features, geometry)
        features = torch.einsum("ij,zaj->zai", (o3.irreducible_basis_to_xyz(), features))
        return features


EPSILON = 1e-8
rbf_high = 2.0


def accelerations(points, masses=None):
    """
    inputs:
    -points: a list of 3-tuples of point coordinates
    -masses: a list (of equal length N) of masses

    returns:
    -shape [N, 3] numpy array of accelerations under Newtonian gravity
    """
    accels = []
    if masses is None:
        masses = [1.0 for _ in range(len(points))]
    for ri_ in points:
        accel_vec = np.array((0., 0., 0.))
        for rj_, m in zip(points, masses):
            rij_ = ri_ - rj_
            dij_ = np.linalg.norm(rij_)
            if (ri_ != rj_).any():
                accel_update = -rij_ / (np.power(dij_, 3) + EPSILON) * m
                accel_vec += accel_update
        accels.append(accel_vec)
    assert len(accels) == len(points)
    return np.array(accels)


def random_points_and_masses(max_points=10, min_mass=0.5, max_mass=2.0,
                             max_coord=rbf_high, min_separation=0.5):
    """
    returns:
    -shape [N, 3] numpy array of points, where N is between 2 and max_points
    -shape [N] numpy array of masses
    """
    num_points = random.randint(2, max_points)
    candidate_points = []
    for point in range(num_points):
        candidate_points.append(
            np.array([random.uniform(-max_coord, max_coord) for _ in range(3)]))

    # remove points that are closer than min_separation
    output_points = []
    for point in candidate_points:
        include_point = True
        for previous_point in output_points:
            if np.linalg.norm(point - previous_point) < min_separation:
                include_point = False
        if include_point:
            output_points.append(point)

    points_ = np.array(output_points)
    masses_ = np.random.rand(len(output_points)) * (max_mass - min_mass) + min_mass
    return points_, masses_


def train(net):
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    max_steps = 301
    validation_size = 10
    print_freq = 25

    for step in range(max_steps):
        points, masses = random_points_and_masses(10)
        accels = accelerations(points, masses)

        points = torch.from_numpy(points).reshape(1, -1, 3)
        masses = torch.from_numpy(masses).reshape(1, -1, 1)
        accels = torch.from_numpy(accels).reshape(1, -1, 3)
        output = net(masses, points)  # [3, N]

        # spherical harmonics are given in y,z,x order
        loss = torch.mean((output - accels)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = loss.item()

        if step % print_freq == 0:
            for _ in range(validation_size):
                val_points, val_masses = random_points_and_masses(50)
                val_accels = accelerations(val_points, val_masses)

                val_points = torch.from_numpy(val_points).reshape(1, -1, 3)
                val_masses = torch.from_numpy(val_masses).reshape(1, -1, 1)
                val_accels = torch.from_numpy(val_accels).reshape(1, -1, 3)

                output = net(val_masses, val_points)
                output = torch.transpose(output, 0, 1)

                # spherical harmonics are given in y,z,x order
                output = output @ o3.irreducible_basis_to_xyz().t()

                loss = torch.mean((output - val_accels)**2)
            print('Step {0}: validation loss = {1}'.format(step, loss.item()))

            plt.figure()
            x = np.linspace(0, 2.0, 100)
            y = net.conv.kernel.R(torch.from_numpy(x)).detach().numpy()
            plt.plot(x, y)
            start = 25
            plt.plot(x[start:], -1 / (x[start:]**2))
            plt.savefig('validation_result_{0}.png'.format(step))

        if step % 500 == 0:
            print('Step {0}, Loss {1}'.format(step, step_loss))
    return net


def main():
    net = GravityNet()
    train(net)


main()
