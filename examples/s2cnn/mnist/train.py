import gzip
import math
import pickle

import numpy as np
import torch
from e3nn import o3
from e3nn.nn import SO3Activation


def s2_near_identity_grid(max_beta=math.pi / 8, n_alpha=8, n_beta=3):
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_near_identity_grid(max_beta=math.pi / 8, max_gamma=2 * math.pi, n_alpha=8, n_beta=3, n_gamma=None):
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid):
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid):
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class S2ConvNet_original(torch.nn.Module):
    def __init__(self):
        super().__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 60
        lmax1 = 10

        b_l1 = 10
        lmax2 = 5

        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.from_s2 = o3.FromS2Grid((b_in, b_in), lmax1)

        self.conv1 = S2Convolution(1, f1, lmax1, kernel_grid=grid_s2)

        self.act1 = SO3Activation(lmax1, lmax2, torch.relu, b_l1)

        self.conv2 = SO3Convolution(f1, f2, lmax2, kernel_grid=grid_so3)

        self.act2 = SO3Activation(lmax2, 0, torch.relu, b_l2)

        self.w_out = torch.nn.Parameter(torch.randn(f2, f_output))

    def forward(self, x):
        x = x.transpose(-1, -2)  # [batch, features, alpha, beta] -> [batch, features, beta, alpha]
        x = self.from_s2(x)  # [batch, features, beta, alpha] -> [batch, features, irreps]
        x = self.conv1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.conv2(x)  # [batch, features, irreps] -> [batch, features, irreps]
        x = self.act2(x)  # [batch, features, scalar]
        x = x.flatten(1) @ self.w_out / self.w_out.shape[0]

        return x


MNIST_PATH = "s2_mnist.gz"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-3


def load_data(path, batch_size):

    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(dataset["train"]["labels"].astype(np.int64))

    # train_data /= 57  This normalization was hurtful, see @dmklee comment in discussions/344

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(dataset["test"]["labels"].astype(np.int64))

    # test_data /= 57

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


def main():
    train_loader, test_loader, train_dataset, _ = load_data(MNIST_PATH, BATCH_SIZE)

    classifier = S2ConvNet_original()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            print(
                "\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                    epoch + 1, NUM_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.item()
                ),
                end="",
            )
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print("Test Accuracy: {0}".format(100 * correct / total))


if __name__ == "__main__":
    main()
