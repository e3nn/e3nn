import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.util import get_mask

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
        mask = get_mask.get_mask((y_true.size(0), y_true.size(-3), y_true.size(-2), y_true.size(-1)), valid)

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


    loss_per_voxel = F.cross_entropy(
        y_pred_f, y_true_f, reduce=False, weight=class_weight).view(y_true.shape).squeeze()

    if valid is not None:
        mask = get_mask.get_mask(loss_per_voxel.shape, valid)

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