# pylint: disable=no-member, not-callable, missing-docstring, line-too-long, invalid-name, logging-format-interpolation
import copy
import importlib.machinery
import logging
import os
import shutil
import time
import types

import torch
import torch.nn as nn

from e3nn.util.dataset.molecules import (QM9, VoxelizeBlobs,
                                           center_positions,
                                           random_rotate_translate)


def main(log_dir, model_path, dataset, batch_size, learning_rate, num_workers, restore_dir, lr_value, lr_steps):
    arguments = copy.deepcopy(locals())

    os.mkdir(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2(model_path, os.path.join(log_dir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0")

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(5, 1).to(device)
    perceptron = nn.Linear(5, 1).to(device)
    with torch.no_grad():
        # perceptron for QM9
        perceptron.weight.copy_(perceptron.weight.new_tensor([[-38.0770, -0.6040, -75.2287, -54.7525, -99.8718]]))
        perceptron.bias.copy_(perceptron.bias.new_tensor([0.0292]))

    if restore_dir is not None:
        model.load_state_dict(torch.load(os.path.join(restore_dir, "state.pkl")))
        perceptron.load_state_dict(torch.load(os.path.join(restore_dir, "perceptron.pkl")))

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    # Load the dataset
    render = VoxelizeBlobs(5, model.input_size, 0.2 * 77 / model.input_size)
    def transform(positions, qualias, energy):
        positions = random_rotate_translate(center_positions(positions))
        x = render(positions, qualias)
        return x, torch.tensor(energy, dtype=torch.float32)

    train_set = QM9("qm9_data", transform=transform)
    torch.manual_seed(5)
    n = len(train_set)
    indices = torch.randperm(n)
    if dataset == "train":
        indices = indices[:n // 2]
    elif dataset == "val":
        indices = indices[n // 2: 7 * n // 10]
    elif dataset == "test":
        indices = indices[7 * n // 10:]
    train_set = torch.utils.data.Subset(train_set, indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0)

    def train_step(data, target):
        model.train()
        data, target = data.to(device), target.to(device)

        rough = perceptron(data.view(data.size(0), data.size(1), -1).sum(-1))
        prediction = rough + 0.04 * model(data)

        loss = (prediction.view(-1) - target.view(-1)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.sqrt().item()

    def get_learning_rate(epoch):
        assert len(lr_value) == len(lr_steps) + 1
        for lim, lr in zip(lr_steps, lr_value):
            if epoch < lim:
                return lr * learning_rate
        return lr_value[-1] * learning_rate

    dynamics = []
    epoch = 0

    if restore_dir is not None:
        dynamics = torch.load(os.path.join(restore_dir, "dynamics.pkl"))
        epoch = dynamics[-1]['epoch'] + 1

    for epoch in range(epoch, 2000):

        lr = get_learning_rate(epoch)
        logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss = train_step(data, target)

            total_loss += loss

            avg_loss = total_loss / (batch_idx + 1)

            logger.info("[{}:{}/{}] RMSE={:.2} <RMSE>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, avg_loss,
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step,
            ))
            time_before_load = time.perf_counter()

            dynamics.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'step': epoch * len(train_loader) + batch_idx,
                'learning_rate': lr,
                'batch_size': len(data),
                'loss': loss,
                'avg_loss': avg_loss,
            })

        torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
        torch.save(perceptron.state_dict(), os.path.join(log_dir, "perceptron.pkl"))
        torch.save(dynamics, os.path.join(log_dir, "dynamics.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="train")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--lr_value", type=float, nargs='+', default=[1, 0.1, 0.01])
    parser.add_argument("--lr_steps", type=int, nargs='+', default=[500, 1000])
    parser.add_argument("--restore_dir", type=str)

    args = parser.parse_args()

    main(**args.__dict__)
