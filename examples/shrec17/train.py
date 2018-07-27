# pylint: disable=E1101,R,C,W1202
import torch
import torch.nn.functional as F

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np

from se3cnn.util.dataset.shapes import Shrec17, CacheNPY, Obj2Voxel, EqSampler


def main(log_dir, model_path, augmentation, dataset, batch_size, learning_rate, num_workers):
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

    model = mod.Model(55).to(device)

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    # Load the dataset
    # Increasing `repeat` will generate more cached files
    cache = CacheNPY("v64d", transform=Obj2Voxel(64, double=True, rotate=True), repeat=augmentation)
    def transform(x):
        x = cache(x)
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0) / 8

    def target_transform(x):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        return classes.index(x[0])

    train_set = Shrec17("shrec17_data", dataset, perturbed=True, download=True, transform=transform, target_transform=target_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=EqSampler(train_set), num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0)

    def train_step(data, target):
        model.train()
        data, target = data.to(device), target.to(device)

        prediction = model(data)
        loss = F.cross_entropy(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = prediction.argmax(1).eq(target).long().sum().item()

        return loss.item(), correct

    def get_learning_rate(epoch):
        limits = [500, 1000]
        lrs = [1, 0.1, 0.01]
        assert len(lrs) == len(limits) + 1
        for lim, lr in zip(limits, lrs):
            if epoch < lim:
                return lr * learning_rate
        return lrs[-1] * learning_rate

    dynamics = []

    for epoch in range(2000):

        lr = get_learning_rate(epoch)
        logger.info("learning rate = {} and batch size = {}".format(lr, train_loader.batch_size))
        for p in optimizer.param_groups:
            p['lr'] = lr

        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, correct = train_step(data, target)

            total_loss += loss
            total_correct += correct

            avg_loss = total_loss / (batch_idx + 1)
            avg_correct = total_correct / len(data) / (batch_idx + 1)

            logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, avg_loss,
                correct / len(data), avg_correct,
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()

            dynamics.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'step': epoch * len(train_loader) + batch_idx,
                'learning_rate': lr,
                'batch_size': len(data),
                'loss': loss,
                'correct': correct,
                'avg_loss': avg_loss,
                'avg_correct': avg_correct,
            })

        torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
        torch.save(dynamics, os.path.join(log_dir, "dynamics.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.5)

    args = parser.parse_args()

    main(**args.__dict__)