# pylint: disable=C,R,E1101,not-callable
import sys, os
import torch
import se3cnn
import numpy as np
import scipy

from se3cnn.utils import torch_default_dtype
import se3cnn.point_utils as point_utils
from se3cnn.non_linearities import ScalarActivation
from se3cnn.convolution import SE3PointNeighborConvolution
from se3cnn.point_kernel import SE3PointKernel
from se3cnn.blocks.point_gated_block_mod import PointGatedBlock

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
            'radii': torch.linspace(0, max_radius, steps=num_radial, dtype=torch.float64),
            'activation': (torch.nn.functional.relu, torch.sigmoid),
            'kernel': SE3PointKernel
        }

        self.layers = torch.nn.ModuleList([PointGatedBlock(features[i],
                                                           features[i+1],
                                                           convolution=SE3PointNeighborConvolution,
                                                           **kwargs) for i in
                                           range(len(features) - 1)])
        with torch_default_dtype(torch.float64):
            self.layers.extend([AvgSpacial(), torch.nn.Dropout(p=0.2), torch.nn.Linear(64, num_classes)])

    def forward(self, input, coords=None, neighbor=None,
                relative_mask=None):
        if coords is None or neighbor is None:
            raise ValueError()
        output = input
        for i in range(self.num_features - 1):
            conv = self.layers[i]
            output = conv(output, coords=coords, neighbors=neighbor,
                          relative_mask=relative_mask)
        for i in range(self.num_features - 1, len(self.layers)):
            if i == self.num_features - 1:
                output = output[..., :4]
            layer = self.layers[i]
            output = layer(output)  # Ignore last dummy point
        return output

def get_dataset():
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 0, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0), (0, 0, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 0)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 0, 0)],  # L 
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 0, 0)],  # T
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (0, 0, 0)]]  # zigzag 
    masks = [[1, 1, 1, 1, 0] for i in range(len(tetris))]
    neighbors = [[(1, 2, 4), (0, 4, 4), (0, 3, 4), (2, 4, 4), (4, 4, 4)],  # chiral_shape_1
                 [(1, 2, 4), (0, 4, 4), (0, 3, 4), (2, 4, 4), (4, 4, 4)],  # chiral_shape_2
                 [(1, 2, 4), (0, 3, 4), (0, 3, 4), (1, 2, 4), (4, 4, 4)],  # square
                 [(1, 4, 4), (0, 2, 4), (1, 3, 4), (2, 4, 4), (4, 4, 4)],  # line
                 [(1, 2, 3), (0, 4, 4), (0, 4, 4), (0, 4, 4), (4, 4, 4)],  # corner
                 [(1, 3, 4), (0, 2, 4), (1, 4, 4), (0, 4, 4), (4, 4, 4)],  # L
                 [(1, 4, 4), (0, 2, 3), (1, 4, 4), (1, 4, 4), (4, 4, 4)],  # T
                 [(1, 4, 4), (0, 2, 4), (1, 3, 4), (2, 4, 4), (4, 4, 4)]]  # zigzag
    tetris = torch.tensor(tetris, dtype=torch.float64)
    masks = torch.tensor(masks, dtype=torch.float64)
    neighbors = torch.LongTensor(neighbors)
    batch, N, K = neighbors.shape
    relative_masks = masks[torch.arange(0, batch,
                                        dtype=torch.long).reshape(-1, 1, 1),
                           neighbors]
    labels = torch.arange(len(tetris))

    return tetris, neighbors, relative_masks, labels

def train(net, coords, neighbors, relative_masks, labels):
    net.train()

    loss_fn = torch.nn.CrossEntropyLoss()

    max_epochs = 250
    optimizer = torch.optim.Adam(net.parameters(), lr=1.2, weight_decay=1e-5)

    batch, N, K = neighbors.size()

    for epoch in range(max_epochs):
        input = torch.ones(batch, 1, N, dtype=torch.float64)
        input[...,-1] = 0.
        predictions = net(input, coords=coords, neighbor=neighbors,
                          relative_mask=relative_masks)
        losses = loss_fn(predictions, labels)
        loss = losses.mean()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        argmax = predictions.argmax(1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('Epoch {0}, Loss {1}, Acc {2}'.format(epoch, loss, acc))
    return net

def test(net, tetris, neighbors, relative_masks):
    net.eval()

    test_set_size = 25
    rng = np.random.RandomState()

    correct_predictions = 0
    total_predictions = 0
    for _ in range(test_set_size):
        for label, (shape, neighbor, relative_mask) in enumerate(zip(tetris,
                                                                     neighbors,
                                                                     relative_masks)):
            rotation = random_rotation_matrix(rng)
            rotated_shape = np.dot(shape, rotation)
            translation = np.expand_dims(np.random.uniform(low=-3., high=3., size=(3)), axis=0)
            translated_shape = torch.from_numpy(rotated_shape + translation).unsqueeze(-3)
            batch = 1
            N, _ = shape.size()
            neighbor = neighbor.unsqueeze(0)
            relative_mask = relative_mask.unsqueeze(0)
            input = torch.ones(batch, 1, N, dtype=torch.float64)
            input[..., -1] = 0.
            prediction = net(input, coords=translated_shape,
                             neighbor=neighbor,
                             relative_mask=relative_mask).argmax(1)

            correct_predictions += (prediction == label).item()
            total_predictions += 1
    print('Test Accuracy: {0}'.format(correct_predictions / total_predictions))

def main(argv):
    tetris, neighbors, relative_masks, labels = get_dataset()

    net = SE3Net(8)
    train(net, tetris, neighbors, relative_masks, labels)
    test_epochs = 10
    for _ in range(test_epochs):
        test(net, tetris, neighbors, relative_masks)

if __name__ == '__main__':
    main(sys.argv[1:])
