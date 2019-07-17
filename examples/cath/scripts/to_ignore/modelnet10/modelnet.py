# pylint: disable=E1101,R,C,E1102
import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from se3cnn import basis_kernels
from functools import partial
from se3cnn.blocks import GatedBlock
from experiments.datasets.modelnet.modelnet_old import ModelNet10, Obj2Voxel, CacheNPY
from sklearn.model_selection import StratifiedKFold
import itertools
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


class _StratifiedKFold_Dataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class StratifiedKFold_Dataset:
    def __init__(self, labels, n_splits):
        self.splits = list(
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True
            ).split(
                np.zeros(len(labels)),
                labels
            )
        )
        self.n_splits = n_splits

    def split_trainset(self, dataset, i):
        return _StratifiedKFold_Dataset(dataset, self.splits[i][0])

    def split_testset(self, dataset, i):
        return _StratifiedKFold_Dataset(dataset, self.splits[i][1])


def compose(t1, t2):
    def f(x):
        return t2(t1(x))
    return f


def to_tensor(x):
    x = np.array(x).astype(np.uint8)
    x = torch.tensor(x, dtype=torch.float)
    x = x.unsqueeze(1 if x.dim() == 4 else 0)
    x = x / 8
    return x


classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]


def target_transform(x):
    return classes.index(x)


cache = CacheNPY("azr32ddxy",
                 repeat=12,
                 transform=Obj2Voxel(32,
                                     rotate=False,
                                     zrotate=True,
                                     double=True,
                                     diagonal_bounding_box_xy=True))
setAZR = ModelNet10("./root/", mode='train_full',
                    classes=classes,
                    download=True,
                    transform=compose(cache, to_tensor),
                    target_transform=target_transform)

cache = CacheNPY("azr32ddxy",
                 repeat=12,
                 pick_randomly=False,
                 transform=Obj2Voxel(32,
                                     rotate=False,
                                     zrotate=True,
                                     double=True,
                                     diagonal_bounding_box_xy=True))
setAZRE = ModelNet10("./root/", mode='train_full',
                     classes=classes,
                     download=True,
                     transform=compose(cache, to_tensor),
                     target_transform=target_transform)

for _ in range(12):
    list(torch.utils.data.DataLoader(setAZR, batch_size=16, num_workers=12))


skf = StratifiedKFold_Dataset([y for x, y in setAZR], 4)


def plot_repr(x):
    plt.imshow(x[0].detach().cpu().numpy().mean(-1))


class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (1, ),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (200, )
        ]

        radial_window = partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                mode='compromise', border_dist=0, sigma=0.6)

        common_block_params = {
            'size': 5,
            'stride': 2,
            'padding': 3,
            'normalization': 'batch',
            'radial_window': radial_window,
            'capsule_dropout_p': 0.1
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            nn.Linear(features[-1][0], 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, features, x, y, z]
        '''
        x = self.sequence(x)  # [batch, features]

        return x


def train(model, dataset, n_epoch):
    batch_size = 64

    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

    data = []

    for epoch in range(n_epoch):

        if epoch in [100, 200]:
            for pg in optimizer.param_groups:
                pg['lr'] /= 3

        for i, (input, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # forward and backward propagation
            output = model(input)
            loss = F.nll_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # download results on the CPU
            loss = loss.detach().cpu().item()
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            # compute the accuracy
            acc = float(np.sum(output.argmax(-1) == target) / target.size)

            data.append({
                "epoch": epoch,
                "i": i / len(dataloader),
                "loss": loss,
                "accuracy": acc
            })

            print("{}:{}/{}: acc={}% loss={}".format(epoch, i, len(dataloader), 100 * acc, loss))
    return data


def test(model, dataset):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

    correct = 0
    for i, (input, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        if input.dim() == 6:  # eval augmentation
            an = input.size(1)
            input = input.view(-1, *input.size()[2:])
        else:
            an = None

        # forward and backward propagation
        output = model(input)

        if an:
            output = output.view(-1, an, *output.size()[1:])
            output = output.mean(1)

        # download results on the CPU
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # compute the accuracy
        correct += np.sum(output.argmax(-1) == target)

        print("{}/{}".format(i, len(dataloader)))

    return correct / len(dataset)


def confusion_matrix(model, dataset):
    from sklearn.metrics import confusion_matrix

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

    y_true = []
    y_pred = []

    for input, target in dataloader:
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        if input.dim() == 6:  # eval augmentation
            an = input.size(1)
            input = input.view(-1, *input.size()[2:])
        else:
            an = None

        # forward and backward propagation
        output = model(input)

        if an:
            output = output.view(-1, an, *output.size()[1:])
            output = output.mean(1)

        # download results on the CPU
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        y_true += list(target)
        y_pred += list(output.argmax(-1))

    return confusion_matrix(y_true, y_pred)


def plot(data):
    plt.plot([x['epoch'] + x['i'] for x in data], [x['loss'] for x in data])
    plt.yscale('log')


results = []

for i in range(skf.n_splits):
    model = CNN()
    if torch.cuda.is_available():
        model.cuda()

    data = train(model, skf.split_trainset(setAZR, i), 300)
    plot(data)

    results.append((
        test(model, skf.split_trainset(setAZRE, i)),
        test(model, skf.split_testset(setAZRE, i))))

    print(results)

results = np.array(results)
print(results)
print(results.mean(0))


plt.figure()
cm = confusion_matrix(model, skf.split_testset(setAZRE, i))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion_matrix.pdf")
