"""Classify tetris using gate activation function

Implement a equivariant model using gates to fit the tetris dataset
Exact equivariance to :math:`E(3)`

>>> test()
"""
import torch
from torch_geometric.data import Data, DataLoader

from e3nn import o3
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork


def tetris():
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = torch.tensor(
        [
            [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
            [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
            [0, 1, 0, 0, 0, 0, 0],  # square
            [0, 0, 1, 0, 0, 0, 0],  # line
            [0, 0, 0, 1, 0, 0, 0],  # corner
            [0, 0, 0, 0, 1, 0, 0],  # L
            [0, 0, 0, 0, 0, 1, 0],  # T
            [0, 0, 0, 0, 0, 0, 1],  # zigzag
        ],
        dtype=torch.get_default_dtype(),
    )

    # apply random rotation
    pos = torch.einsum("zij,zaj->zai", o3.rand_matrix(len(pos)), pos)

    return pos, labels


def make_batch(pos):
    # put in torch_geometric format
    dataset = [Data(pos=pos, x=torch.ones(4, 1)) for pos in pos]
    return next(iter(DataLoader(dataset, batch_size=len(dataset))))


def Network():
    return SimpleNetwork(
        irreps_in="0e",
        irreps_out="0o + 6x0e",
        max_radius=1.5,
        num_neighbors=2.0,
        num_nodes=4.0,
    )


def main():
    x, y = tetris()
    train_x, train_y = make_batch(x[1:]), y[1:]  # dont train on both chiral shapes

    x, y = tetris()
    test_x, test_y = make_batch(x), y

    f = Network()

    print("Built a model:")
    print(f)

    optim = torch.optim.Adam(f.parameters(), lr=1e-3)

    # == Training ==
    for step in range(300):
        pred = f(train_x)
        loss = (pred - train_y).pow(2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 10 == 0:
            accuracy = f(test_x).round().eq(test_y).all(dim=1).double().mean(dim=0).item()
            print(f"epoch {step:5d} | loss {loss:<10.1f} | {100 * accuracy:5.1f}% accuracy")

    # == Check equivariance ==
    # Because the model outputs (psuedo)scalars, we can easily directly
    # check its equivariance to the same data with new rotations:
    print("Testing equivariance directly...")
    rotated_x, _ = tetris()
    rotated_x = make_batch(rotated_x)
    error = f(rotated_x) - f(test_x)
    print(f"Equivariance error = {error.abs().max().item():.1e}")


if __name__ == "__main__":
    main()


def test():
    torch.set_default_dtype(torch.float64)

    data, labels = tetris()
    data = make_batch(data)
    f = Network()

    pred = f(data)
    loss = (pred - labels).pow(2).sum()
    loss.backward()

    rotated_data, _ = tetris()
    rotated_data = make_batch(rotated_data)
    error = f(rotated_data) - f(data)
    assert error.abs().max() < 1e-10


def profile():
    data, labels = tetris()
    data = make_batch(data)
    data = data.to(device="cuda")
    labels = labels.to(device="cuda")

    f = Network()
    f.to(device="cuda")

    optim = torch.optim.Adam(f.parameters(), lr=1e-2)

    called_num = [0]

    def trace_handler(p):
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        p.export_chrome_trace("test_trace_" + str(called_num[0]) + ".json")
        called_num[0] += 1

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=50, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as p:
        for _ in range(52):
            pred = f(data)
            loss = (pred - labels).pow(2).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

            p.step()
