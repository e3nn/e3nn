# pylint: disable=C,R,E1101,not-callable
import sys, os
import torch
import se3cnn
import numpy as np
import scipy

from se3cnn.utils import torch_default_dtype
import se3cnn.point_utils as point_utils
from se3cnn.convolution import SE3PointConvolution

torch.set_default_dtype(torch.float64)

EPSILON = 1e-8


def random_rotation_matrix(numpy_random_state):
    """
    Generates a random 3D rotation matrix from axis and angle.
    Args:
        numpy_random_state: numpy random state object
    Returns:
        Random rotation matrix.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


size = 3  # To match image case


class SE3Net(torch.nn.Module):
    def __init__(self, num_classes, num_radial=size // 2 + 1, max_radius=size // 2):
        super(SE3Net, self).__init__()

        features = [(1,), (2, 2, 2, 1), (4, 4, 4, 4), (6, 4, 4, 0), (64,)]
        self.num_features = len(features)

        kwargs = {
            'radii': torch.linspace(0, max_radius, steps=num_radial,
                                    dtype=torch.float64),
        }

        self.layers = torch.nn.ModuleList([])
        for i in range(len(features) - 1):
            Rs_in = list(zip(features[i],
                             range(len(features[i]))))
            Rs_out = list(zip(features[i + 1],
                              range(len(features[i + 1]))))
            self.layers.append(
                SE3PointConvolution(Rs_in, Rs_out, **kwargs)
            )
        with torch_default_dtype(torch.float64):
            self.layers.extend([AvgSpacial(), torch.nn.Dropout(p=0.2), torch.nn.Linear(64, num_classes)])

    def forward(self, input, difference_mat):
        output = input
        for i in range(self.num_features - 1):
            conv = self.layers[i]
            output = conv(output, difference_mat)
        for i in range(self.num_features - 1, len(self.layers)):
            layer = self.layers[i]
            output = layer(output)
        return output

def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L
    tetris = torch.tensor(tetris, dtype=torch.float64)
    labels = torch.arange(len(tetris))

    return tetris, labels

def train(net, diff_M, labels):
    net.train()

    loss_fn = torch.nn.CrossEntropyLoss()

    max_epochs = 250
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-2, weight_decay=1e-5)

    batch, N, _M, _ = diff_M.size()

    for epoch in range(max_epochs):
        input = torch.ones(batch, 1, N, dtype=torch.float64)
        predictions = net(input, diff_M)
        losses = loss_fn(predictions, labels)
        loss = losses.mean()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        argmax = predictions.argmax(1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('Epoch {0}, Loss {1}, Acc {2}'.format(epoch, loss, acc))
    return net

def test(net, tetris):
    net.eval()

    test_set_size = 25
    rng = np.random.RandomState()

    correct_predictions = 0
    total_predictions = 0
    for _ in range(test_set_size):
        for label, shape in enumerate(tetris):
            rotation = random_rotation_matrix(rng)
            rotated_shape = np.dot(shape, rotation)
            translation = np.expand_dims(np.random.uniform(low=-3., high=3., size=(3)), axis=0)
            translated_shape = torch.from_numpy(rotated_shape + translation).unsqueeze(-3)
            diff_M = point_utils.difference_matrix(translated_shape)
            batch, N, _M, _ = diff_M.size()
            input = torch.ones(batch, 1, N, dtype=torch.float64)
            prediction = net(input, diff_M).argmax(1)

            correct_predictions += (prediction == label).item()
            total_predictions += 1
    print('Test Accuracy: {0}'.format(correct_predictions / total_predictions))

def main(argv):
    tetris, labels = get_dataset()
    diff_M = point_utils.difference_matrix(tetris)

    net = SE3Net(8)
    train(net, diff_M, labels)
    test_epochs = 10
    for _ in range(test_epochs):
        test(net, tetris)

if __name__ == '__main__':
    main(sys.argv[1:])
