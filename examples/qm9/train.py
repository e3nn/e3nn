# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np

from se3cnn.util.dataset.molecules import QM9, center_positions, random_rotate_translate, VoxelizeBlobs


def main(log_dir, model_path, dataset, batch_size, learning_rate, num_workers, restore_dir):
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
        perceptron.weight.copy_(perceptron.weight.new_tensor([[-38.0770,  -0.6040, -75.2287, -54.7525, -99.8718]]))
        perceptron.bias.copy_(perceptron.bias.new_tensor([0.0292]))

    if restore_dir is not None:
        model.load_state_dict(torch.load(os.path.join(restore_dir, "state.pkl")))
        perceptron.load_state_dict(torch.load(os.path.join(restore_dir, "perceptron.pkl")))

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    # Load the dataset
    render = VoxelizeBlobs(5, 77, 0.2)
    def transform(positions, qualias, energy):
        positions = random_rotate_translate(center_positions(positions))
        x = render(positions, qualias)
        return x, torch.tensor(energy, dtype=torch.float32)

    train_set = QM9("qm9_data", transform=transform)
    torch.manual_seed(5)
    indices = torch.randperm(len(train_set))
    train_set = torch.utils.data.Subset(train_set, indices[:len(train_set) // 2])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(perceptron.parameters()), lr=0)

    def train_step(data, target):
        model.train()
        data, target = data.to(device), target.to(device)

        prediction = model(data) + perceptron(data.view(data.size(0), data.size(1), -1).sum(-1))
        loss = (prediction - target.view(-1)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.sqrt().item()

    def get_learning_rate(epoch):
        limits = [500, 1000]
        lrs = [1, 0.1, 0.01]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate

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

            logger.info("[{}:{}/{}] RMSE={:.2} <RMSE>={:.2} time={:.2}+{:.2} perceptron={}+{}".format(
                epoch, batch_idx, len(train_loader),
                loss, avg_loss,
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step,
                perceptron.weight, perceptron.bias.item()))
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
    parser.add_argument("--restore_dir", type=str)

    args = parser.parse_args()

    main(**args.__dict__)

