# pylint: disable=C,R,E1101
'''
Architecture for MRI image segmentation.

'''

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import time

from se3_cnn.blocks import GatedBlock
from se3_cnn.datasets import MRISegmentation


class FlattenSpacial(nn.Module):
    def __init__(self):
        super(FlattenSpacial, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1, x.size(-3)*x.size(-2)*x.size(-1))


class Model(nn.Module):

    def __init__(self, output_size, filter_size=5):
        super(Model, self).__init__()

        features = [(1,),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    (output_size,)]

        from se3_cnn import basis_kernels
        radial_window_dict = {
            'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
            'radial_window_fct_kwargs': {
                'mode': 'compromise',
                'border_dist': 0.,
                'sigma': .6
            }
        }
        common_block_params = {
            'size': filter_size,
            'padding': filter_size//2,
            'stride': 1,
            'batch_norm_before_conv': False,
            'radial_window_dict': radial_window_dict
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1],
                             **common_block_params, **block_params[i])
                  for i in range(len(block_params))]

        self.layers = torch.nn.Sequential(
            *blocks,
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def dice_coefficient_orig_binary(y_pred, y_true, epsilon=1e-5):
    """As originally specified: using binary vectors and an explicit average
       over classes """

    y_pred = nn.Softmax(dim=1)(y_pred)

    dice_coeff = 0
    for i in range(y_pred.size(1)):

        # Convert to binary values
        y_pred_b = torch.max(y_pred, dim=1)[1] == i
        y_true_b = y_true == i

        y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1).float()
        y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()

        s1 = y_true_f
        s2 = y_pred_f

        # Calculate dice score
        dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
                      (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))

    dice_coeff /= float(y_pred.size(1))

    return dice_coeff.mean()


def dice_coefficient_orig(y_pred, y_true, epsilon=1e-5):
    """Original version but multiplying probs instead of 0-1 variables"""

    y_pred = nn.Softmax(dim=1)(y_pred)

    dice_coeff = 0
    for i in range(y_pred.size(1)):

        y_pred_b = y_pred[:,i,:,:,:]
        y_true_b = y_true == i

        y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1)
        y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()

        s1 = y_true_f
        s2 = y_pred_f

        dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
                      (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))

    dice_coeff /= float(y_pred.size(1))

    return dice_coeff.mean()


def dice_coefficient_onehot(y_pred, y_true, epsilon=1e-5):
    """Reimplementation with matrix operations - with onehot encoding
       of y_true"""

    y_pred = nn.Softmax(dim=1)(y_pred)

    y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true_f = y_true.view(y_true.size(0), y_true.size(1), -1)

    intersection = torch.sum(y_true_f * y_pred_f, dim=2)
    coeff = 2./y_pred.shape[1] * torch.sum(
        intersection / (epsilon +
                        torch.sum(y_true_f, dim=2) +
                        torch.sum(y_pred_f, dim=2)),
        dim=1)
    return coeff.mean()


def dice_coefficient(y_pred, y_true, epsilon=1e-5):
    """Reimplementation with matrix operations - directly on y_true class
       labels"""

    y_pred = nn.Softmax(dim=1)(y_pred)

    y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true_f = y_true.view(y_true.size(0), -1)

    all_classes = torch.autograd.Variable(
        torch.LongTensor(np.arange(y_pred_f.size(1))))
    if torch.cuda.is_available():
        all_classes = all_classes.cuda()
    coefs = []
    for i in range(y_pred.shape[0]):
        # Dynamically create one-hot encoding
        # TODO: is this more memory efficient?
        class_at_voxel = (all_classes.view(-1, 1) == y_true_f[i]).float()
        intersection = torch.sum(class_at_voxel * y_pred_f[i],
                                 dim=1)
        coefs.append(2./y_pred.shape[1] * torch.sum(
            intersection / (epsilon +
                            torch.sum(class_at_voxel, dim=1) +
                            torch.sum(y_pred_f[i], dim=1)),
            dim=0))

    return torch.cat(coefs).mean()


def cross_entropy_loss(y_pred, y_true):

    # Reshape into 2D image, which pytorch can handle
    y_true_f = y_true.view(y_true.size(0), y_true.size(2), -1)
    y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), y_pred.size(2), -1)

    return torch.nn.functional.cross_entropy(y_pred_f, y_true_f)


def main(args):

    torch.backends.cudnn.benchmark = True

    train_filter = ["1000_3",
                    "1001_3",
                    "1002_3",
                    "1006_3",
                    "1007_3",
                    "1008_3",
                    "1009_3",
                    "1010_3",
                    "1011_3",
                    "1012_3",
                    "1013_3",
                    "1014_3"
                    ]
    validation_filter = ["1015_3",
                         "1017_3",
                         "1036_3"
                         ]
    test_filter = ["1003_3",
                   "1004_3",
                   "1005_3",
                   "1018_3",
                   "1019_3",
                   "1023_3",
                   "1024_3",
                   "1025_3",
                   "1038_3",
                   "1039_3",
                   "1101_3",
                   "1104_3",
                   "1107_3",
                   "1110_3",
                   "1113_3",
                   "1116_3",
                   "1119_3",
                   "1122_3",
                   "1125_3",
                   "1128_3"]

    # Check that sets are non-overlapping
    assert len(set(validation_filter).intersection(train_filter)) == 0
    assert len(set(test_filter).intersection(train_filter)) == 0

    if args.mode == 'train':
        train_set = MRISegmentation(args.data_filename,
                                    patch_shape=args.patch_size,
                                    filter=train_filter)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   drop_last=True)
        np.set_printoptions(threshold=np.nan)
        print(np.unique(train_set.labels[0]))

    if args.mode in ['train', 'validate']:
        validation_set = MRISegmentation(args.data_filename,
                                         patch_shape=args.patch_size,
                                         filter=validation_filter)
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        pin_memory=False,
                                                        drop_last=False)

    if args.mode == 'test':
        test_set = MRISegmentation(args.data_filename,
                                   patch_shape=args.patch_size,
                                   filter=test_filter)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=False,
                                                  drop_last=False)


    output_size = 135
    model = Model(output_size=output_size)
    if torch.cuda.is_available():
        model.cuda()

    print("The model contains {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_function = None
    if args.loss == "dice":
        loss_function = lambda *x: -dice_coefficient(*x)
    elif args.loss == "dice_onehot":
        loss_function = lambda *x: -dice_coefficient_onehot(*x)
        target_onehot = torch.FloatTensor(
            args.batch_size, output_size,
            args.patch_size, args.patch_size, args.patch_size)
        if torch.cuda.is_available():
            target_onehot = target_onehot.cuda()
        y_onehot = torch.autograd.Variable(target_onehot)
    elif args.loss == "cross_entropy":
        loss_function = cross_entropy_loss

    epoch_start_index = 0
    if args.mode == 'train':

        for epoch in range(epoch_start_index, args.training_epochs):

            for batch_idx, (data, target) in enumerate(train_loader):

                model.train()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

                optimizer.zero_grad()
                out = model(x)

                # # Compare dice implementations
                # print("dice original impl: ",
                #       dice_coefficient_orig(out, y).data.cpu().numpy())
                # print("dice original impl - binary: ",
                #       dice_coefficient_orig_binary(out, y).data.cpu().numpy())
                # target_onehot = torch.FloatTensor(
                #     target.size(0), output_size,
                #     target.size(-3), target.size(-2), target.size(-1))
                # target_onehot.zero_()
                # target_onehot.scatter_(1, target.cpu(), 1)
                # y_onehot = torch.autograd.Variable(target_onehot)
                # if torch.cuda.is_available():
                #     y_onehot = y_onehot.cuda()
                # print("dice new impl - with one hot: ",
                #       dice_coefficient_onehot(out, y_onehot).data.cpu().numpy())
                # print("dice new impl - on class label: ",
                #       dice_coefficient(out, y).data.cpu().numpy())

                time_start = time.perf_counter()
                if args.loss == "dice_onehot":
                    target_onehot.zero_()
                    target_onehot.scatter_(1, target, 1)
                    loss = loss_function(out, y_onehot)
                else:
                    loss = loss_function(out, y)

                loss.backward()
                optimizer.step()

                if args.loss == "cross_entropy":
                    acc = dice_coefficient(out, y).data[0]
                else:
                    acc = -loss.data[0]

                print("[{}:{}/{}] loss={:.4} acc={:.4} time={:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    float(loss.data[0]), acc,
                    time.perf_counter() - time_start))

            # Adjust patch indices at end of each epoch
            train_set.initialize_patch_indices()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-filename", required=True,
                        help="The name of the data file.")
    parser.add_argument("--patch-size", default=64, type=int,
                        help="Size of patches (default: %(default)s)")
    parser.add_argument("--loss", choices=['dice', 'dice_onehot', 'cross_entropy'],
                        default="cross_entropy",
                        help="Which loss function to use(default: %(default)s)")
    parser.add_argument("--mode", choices=['train', 'test', 'validate'],
                        default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=2, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")

    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    main(args=args)


