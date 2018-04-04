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
from functools import partial

from se3_cnn.blocks import GatedBlock
from se3_cnn.datasets import MRISegmentation


class FlattenSpacial(nn.Module):
    def __init__(self):
        super(FlattenSpacial, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1, x.size(-3)*x.size(-2)*x.size(-1))


class Model(nn.Module):

    def __init__(self, output_size, filter_size=7):
        super(Model, self).__init__()

        features = [(1,),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    # (4, 4, 4, 4),
                    (output_size,)]

        from se3_cnn import basis_kernels
        radial_window = partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                     mode='compromise', border_dist=0, sigma=0.6),

        common_block_params = {
            'size': filter_size,
            'padding': filter_size//2,
            'stride': 1,
            'normalization': "instance",
            'radial_window': radial_window
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            # {'activation': (F.relu, F.sigmoid)},
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


def dice_coefficient_orig_binary(y_pred, y_true, y_pred_is_dist=False,
                                 classes=None, epsilon=1e-5, reduce=True):
    """As originally specified: using binary vectors and an explicit average
       over classes. If y_pred_is_dist is false, the classes variable must specify the
       number of classes"""

    if y_pred_is_dist:
        y_pred = torch.max(nn.Softmax(dim=1)(y_pred), dim=1)[1]

    if classes is None:
        classes = y_pred.size(1)

    dice_coeff = 0
    if torch.cuda.is_available():
        intersection = torch.cuda.FloatTensor(y_pred.size(0), classes).fill_(0)
        union = torch.cuda.FloatTensor(y_pred.size(0), classes).fill_(0)
    else:
        intersection = torch.zeros(y_pred.size(0), classes)
        union = torch.zeros(y_pred.size(0), classes)
    if isinstance(y_pred, torch.autograd.Variable):
        intersection = torch.autograd.Variable(intersection)
        union = torch.autograd.Variable(union)
    for i in range(classes):

        # Convert to binary values
        y_pred_b = (y_pred == i)
        y_true_b = (y_true == i)

        y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1).float()
        y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()

        s1 = y_true_f
        s2 = y_pred_f

        # Calculate dice score
        intersection[:,i] = 2. * torch.sum(s1 * s2, dim=1)
        union[:,i] = torch.sum(s1, dim=1) + torch.sum(s2, dim=1)
        dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
                      (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))

    if reduce:
        return torch.mean(torch.sum(intersection, dim=0) /
                          (epsilon+torch.sum(union, dim=0)))
    else:
        return intersection, union


# def dice_coefficient_orig(y_pred, y_true, epsilon=1e-5):
#     """Original version but multiplying probs instead of 0-1 variables"""
#
#     y_pred = nn.Softmax(dim=1)(y_pred)
#
#     dice_coeff = 0
#     for i in range(y_pred.size(1)):
#
#         y_pred_b = y_pred[:,i,:,:,:]
#         y_true_b = y_true == i
#
#         y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1)
#         y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()
#
#         s1 = y_true_f
#         s2 = y_pred_f
#
#         dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
#                       (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))
#
#     dice_coeff /= float(y_pred.size(1))
#
#     return dice_coeff.mean()


# def dice_coefficient_onehot(y_pred, y_true, epsilon=1e-5, reduce=True):
#     """Reimplementation with matrix operations - with onehot encoding
#        of y_true"""
#
#     y_pred = nn.Softmax(dim=1)(y_pred)
#
#     y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
#     y_true_f = y_true.view(y_true.size(0), y_true.size(1), -1)
#
#     intersection = torch.sum(y_true_f * y_pred_f, dim=2)
#     coeff = 2./y_pred.shape[1] * torch.sum(
#         intersection / (epsilon +
#                         torch.sum(y_true_f, dim=2) +
#                         torch.sum(y_pred_f, dim=2)),
#         dim=1)
#     return coeff.mean()


def dice_coefficient(y_pred, y_true, valid=None, reduce=True, epsilon=1e-5):
    """Reimplementation with matrix operations - directly on y_true class
       labels"""

    y_pred = nn.Softmax(dim=1)(y_pred)

    mask = None
    if valid is not None:
        mask = get_mask((y_true.size(0), y_true.size(-3), y_true.size(-2), y_true.size(-1)), valid)

    all_classes = torch.autograd.Variable(
        torch.LongTensor(np.arange(y_pred.size(1))))
    if torch.cuda.is_available():
        all_classes = all_classes.cuda()
    intersections = []
    unions = []
    for i in range(y_pred.shape[0]):

        if mask is not None:
            y_pred_f = y_pred[i][mask[i]].view(y_pred.size(1), -1)
            y_true_f = y_true[i][mask[i]].view(-1)
        else:
            y_pred_f = y_pred[i].view(y_pred.size(1), -1)
            y_true_f = y_true[i].view(-1)

        if len(y_true_f.shape) > 0:
            # Dynamically create one-hot encoding
            class_at_voxel = (all_classes.view(-1, 1) == y_true_f).float()
            intersection = torch.sum(class_at_voxel * y_pred_f,
                                     dim=1)
            intersections.append(2*intersection)
            unions.append(torch.sum(class_at_voxel, dim=1) +
                          torch.sum(y_pred_f, dim=1))

    if len(intersections) > 0:
        intersections = torch.stack(intersections)
        unions = torch.stack(unions)

    if reduce:
        return (torch.mean(torch.sum(intersections, dim=0) /
                           torch.sum(unions, dim=0)))
    else:
        return intersections, unions


def dice_coefficient_loss(y_pred, y_true, valid=None, reduce=True, epsilon=1e-5):
    if reduce:
        return -dice_coefficient(y_pred, y_true, valid, reduce, epsilon)
    else:
        numerator, denominator = dice_coefficient(y_pred, y_true, valid, reduce, epsilon)
        return -numerator, denominator


def cross_entropy_loss(y_pred, y_true, valid=None, reduce=True, class_weight=None):

    # Reshape into 2D image, which pytorch can handle
    y_true_f = y_true.view(y_true.size(0), y_true.size(2), -1)
    y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), y_pred.size(2), -1)


    loss_per_voxel = torch.nn.functional.cross_entropy(
        y_pred_f, y_true_f, reduce=False, weight=class_weight).view(y_true.shape).squeeze()

    if valid is not None:
        mask = get_mask(loss_per_voxel.shape, valid)

        if reduce:
            return loss_per_voxel[mask].mean()
        else:
            loss_per_voxel_sums = []
            loss_per_voxel_norm_consts = []
            for i in range(y_pred.shape[0]):
                loss_per_voxel_masked = loss_per_voxel[i][mask[i]]
                if len(loss_per_voxel_masked.shape) > 0:
                    loss_per_voxel_sums.append(torch.sum(loss_per_voxel_masked))
                    loss_per_voxel_norm_consts.append(torch.LongTensor([loss_per_voxel_masked.shape[0]]))
            return (torch.cat(loss_per_voxel_sums),
                    torch.cat(loss_per_voxel_norm_consts))
    else:
        if reduce:
            return loss_per_voxel.view(-1).mean()
        else:
            return (torch.sum(loss_per_voxel, dim=1),
                    torch.LongTensor([loss_per_voxel.size(0)]).repeat(loss_per_voxel.shape[0]))


def get_mask(image_shape, index):
    if torch.cuda.is_available():
        mask = torch.cuda.ByteTensor(*image_shape).fill_(0)
    else:
        mask = torch.zeros(image_shape).byte()

    for i in range(index.shape[0]):
        if ((index[i, 1, :] - index[i, 0, :]) > 0).all():
            mask[i,
                 index[i, 0, 0]:index[i, 1, 0],
                 index[i, 0, 1]:index[i, 1, 1],
                 index[i, 0, 2]:index[i, 1, 2]] = 1
    return mask


def infer(model, loader, loss_function):
    model.eval()
    losses_numerator = []
    losses_denominator = []
    out_images = []
    for i in range(len(loader.dataset.unpadded_data_shape)):
        out_images.append(np.full(loader.dataset.unpadded_data_shape[i], -1))
    for i, (data, target, img_index, patch_index, valid) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        y = torch.autograd.Variable(target)
        out = model(x)

        _, out_predict = torch.max(out, 1)
        mask = get_mask(out_predict.shape, valid)
        patch_index = patch_index.cpu().numpy()
        for j in range(out.size(0)):
            out_predict_masked = out_predict[j][mask[j]]
            patch_start = patch_index[j,0] + valid[j,0]
            patch_end = patch_start + (valid[j,1]-valid[j,0])
            if (patch_end-patch_start > 0).all():
                out_images[img_index[j]][patch_start[0]:patch_end[0],
                                         patch_start[1]:patch_end[1],
                                         patch_start[2]:patch_end[2]] = out_predict_masked.view((valid[j,1] - valid[j,0]).tolist()).data.cpu().numpy()

        numerator, denominator = loss_function(out, y, valid=valid, reduce=False)
        try:
            numerator = numerator.data
            denominator = denominator.data
        except:
            pass
        losses_numerator.append(numerator.cpu().numpy())
        losses_denominator.append(denominator.cpu().numpy())

        # print(np.mean(np.sum(losses_numerator[-1], axis=0)/np.sum(losses_denominator[-1], axis=0)), loss_function(out, y, valid).data.cpu().numpy())

    # Check that entire image was filled in
    for out_image in out_images:
        assert not (out_image == -1).any()

    losses_numerator = np.concatenate(losses_numerator)
    losses_denominator = np.concatenate(losses_denominator)
    loss = np.mean(np.sum(losses_numerator, axis=0) / np.sum(losses_denominator, axis=0))
    return out_images, loss


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
                                    filter=train_filter,
                                    log10_signal=args.log10_signal)
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
                                         filter=validation_filter,
                                         randomize_patch_offsets=False,
                                         log10_signal=args.log10_signal)
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        pin_memory=False,
                                                        drop_last=False)

    if args.mode == 'test':
        test_set = MRISegmentation(args.data_filename,
                                   patch_shape=args.patch_size,
                                   filter=test_filter,
                                   randomize_patch_offsets=False,
                                   log10_signal=args.log10_signal)
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
    optimizer.zero_grad()

    loss_function = None
    if args.loss == "dice":
        loss_function = dice_coefficient_loss
    elif args.loss == "cross_entropy":
        if args.class_jing:
            class_weight = torch.Tensor(1/train_set.class_count)
            class_weight *= np.sum(train_set.class_count)/len(train_set.class_count)
            if torch.cuda.is_available():
                class_weight = class_weight.cuda()
        else:
            class_weight = None
        loss_function = lambda *x: cross_entropy_loss(*x, class_weight=class_weight)

    epoch_start_index = 0
    if args.mode == 'train':

        for epoch in range(epoch_start_index, args.training_epochs):

            for batch_idx, (data, target, img_index, patch_index, valid) in enumerate(train_loader):

                model.train()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

                out = model(x)

                # # Compare dice implementations
                # print("dice original impl: ",
                #       dice_coefficient_orig(out, y).data.cpu().numpy())
                # print("dice original impl - binary: ",
                #       dice_coefficient_orig_binary(out, y).data.cpu().numpy())
                # print("dice new impl - on class label: ",
                #       dice_coefficient(out, y).data.cpu().numpy())

                time_start = time.perf_counter()
                loss = loss_function(out, y)
                loss.backward()
                if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
                    optimizer.step()
                    optimizer.zero_grad()

                binary_dice_acc = dice_coefficient_orig_binary(out, y, y_pred_is_dist=True).data[0]

                print("[{}:{}/{}] loss={:.4} acc={:.4} time={:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    float(loss.data[0]), binary_dice_acc,
                    time.perf_counter() - time_start))

            validation_ys, validation_loss = infer(model,
                                                   validation_loader,
                                                   loss_function)

            # Calculate binary dice score on predicted images
            numerators = []
            denominators = []
            for i in range(len(validation_set.data)):
                y_true = torch.LongTensor(validation_set.get_original(i)[1])
                y_pred = torch.LongTensor(validation_ys[i])
                if torch.cuda.is_available():
                    y_true = y_true.cuda()
                    y_pred = y_pred.cuda()
                numerator, denominator = dice_coefficient_orig_binary(
                    y_pred.unsqueeze(0),
                    y_true.unsqueeze(0),
                    classes=output_size,
                    reduce=False)
                numerators.append(numerator)
                denominators.append(denominator)
            numerators = torch.cat(numerators)
            denominators = torch.cat(denominators)
            validation_binary_dice_acc = torch.mean(
                torch.sum(numerators, dim=0) /
                (torch.sum(denominators, dim=0)))

            print('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                validation_loss, validation_binary_dice_acc))

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
    parser.add_argument("--log10-signal", action="store_true", default=False,
                        help="Whether to logarithmize the MIR scan signal (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--class-weighting", action='store_true', default=False,
                        help="switches on class weighting, only used in cross entropy loss (default: %(default)s)")

    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    main(args=args)

