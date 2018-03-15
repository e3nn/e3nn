# pylint: disable=C,R,E1101
'''
Architecture to predict the structural categories of proteins according to the CATH
classification (www.cathdb.info).

'''
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import os
import time
from functools import partial

from se3_cnn.blocks import GatedBlock
from se3_cnn.blocks import NormBlock
from se3_cnn import SE3BatchNorm
from se3_cnn import SE3Convolution
from se3_cnn import basis_kernels

from se3_cnn.non_linearities import NormRelu
from se3_cnn.non_linearities import NormSoftplus
from se3_cnn.non_linearities import ScalarActivation
from se3_cnn.non_linearities import GatedActivation

from se3_cnn.util.optimizers_L1L2 import Adam
from se3_cnn.util.lr_schedulers import lr_scheduler_exponential

from se3_cnn.datasets import Cath

tensorflow_available = True
try:
    import tensorflow as tf

    class Logger(object):
        '''From https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard'''

        def __init__(self, log_dir):
            """Create a summary writer logging to log_dir."""
            self.writer = tf.summary.FileWriter(log_dir)

        def scalar_summary(self, tag, value, step):
            """Log a scalar variable."""
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

        def histo_summary(self, tag, values, step, bins=1000):
            """Log a histogram of the tensor of values."""

            # Create a histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill the fields of the histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)
            self.writer.flush()

except:
    tensorflow_available = False


def get_output_shape(input_size, func):
    f = func(torch.autograd.Variable(torch.ones(2, *input_size)))
    return f.size()[1:]


def print_layer(layers, input_shape):
    """"Method for print architecture during model construction"""

    shape = get_output_shape(input_shape, layers)
    print("layer %2d - %20s: %s [output size %s]" % (len(layers), list(layers.named_modules())[-1][0], tuple(shape), "{:,}".format(np.prod(shape))))


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, size=3, stride=1):
        super().__init__()

        channels = [channels_in] + channels_out

        self.layers = []
        for i in range(len(channels) - 1):
            self.layers += [
                nn.Conv3d(channels[i], channels[i + 1],
                          kernel_size=size,
                          padding=size // 2,
                          stride=stride if i == 0 else 1,
                          bias=False),
                nn.BatchNorm3d(channels[i + 1])]
            if (i + 1) < len(channels) - 1:
                self.layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        if len(channels_out) > 1:
            if channels_in == channels_out[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = nn.Sequential(*[
                    nn.Conv3d(channels[0], channels[-1],
                              kernel_size=1,
                              padding=0,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm3d(channels[-1])])
        self.activation = nn.ReLU(inplace=True)

        # initialize
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_normal(module.weight.data)
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.activation(out)
        return out


class SE3GatedResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window_dict=None,
                 batch_norm_momentum=0.01,
                 batch_norm_mode='maximum',
                 batch_norm_before_conv=True,
                 capsule_dropout_p=0.1,
                 scalar_gate_activation=(F.relu, F.sigmoid),
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_gate_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                GatedBlock(reprs[i], reprs[i + 1],
                           size=size, padding=size//2,
                           stride=conv_stride if i == 0 else 1,
                           activation=activation,
                           radial_window_dict=radial_window_dict,
                           batch_norm_momentum=batch_norm_momentum,
                           batch_norm_mode=batch_norm_mode,
                           batch_norm_before_conv=batch_norm_before_conv,
                           capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    GatedBlock(reprs[0], reprs[-1],
                               size=size, padding=size//2,
                               stride=conv_stride,
                               activation=None,
                               radial_window_dict=radial_window_dict,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_mode=batch_norm_mode,
                               batch_norm_before_conv=batch_norm_before_conv,
                               capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            self.activation = GatedActivation(
                repr_in=reprs[-1],
                size=size,
                radial_window_dict=radial_window_dict,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_mode=batch_norm_mode,
                batch_norm_before_conv=batch_norm_before_conv)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        return out


class SE3NormResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window_dict=None,
                 batch_norm_momentum=0.01,
                 batch_norm_mode='maximum',
                 batch_norm_before_conv=True,
                 capsule_dropout_p = 0.1,
                 scalar_activation=F.relu,
                 activation_bias_min=0.5,
                 activation_bias_max=2,
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                NormBlock(reprs[i], reprs[i + 1],
                          size=size, padding=size//2,
                          stride=conv_stride if i == 0 else 1,
                          activation=activation,
                          radial_window_dict=radial_window_dict,
                          batch_norm_momentum=batch_norm_momentum,
                          batch_norm_mode=batch_norm_mode,
                          batch_norm_before_conv=batch_norm_before_conv,
                          capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    NormBlock(reprs[0], reprs[-1],
                              size=size, padding=size//2,
                              stride=conv_stride,
                              activation=None,
                              activation_bias_min=activation_bias_min,
                              activation_bias_max=activation_bias_max,
                              radial_window_dict=radial_window_dict,
                              batch_norm_momentum=batch_norm_momentum,
                              batch_norm_mode=batch_norm_mode,
                              batch_norm_before_conv=batch_norm_before_conv,
                              capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            capsule_dims = [2 * n + 1 for n, mul in enumerate(out_reprs[-1]) for i in
                            range(mul)]  # list of capsule dimensionalities
            self.activation = NormSoftplus(capsule_dims,
                                           scalar_act=scalar_activation,
                                           bias_min=activation_bias_min,
                                           bias_max=activation_bias_max)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        return out


class OuterBlock(nn.Module):
    def __init__(self, in_repr, out_reprs, res_block, size=3, stride=1, **kwargs):
        super().__init__()

        reprs = [[in_repr]] + out_reprs

        self.layers = []
        for i in range(len(reprs) - 1):
            self.layers.append(
                res_block(reprs[i][-1], reprs[i+1],
                          size=size,
                          stride=stride if i == 0 else 1,
                          **kwargs)
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class ResNet(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[block for block in blocks if block is not None])

    def forward(self, x):
        return self.blocks(x)


class ResNet34(ResNet):
    def __init__(self, n_output, size, dense_dropout_p=0.5):

        features = [[ [16]],
                    [ [16] * 2] * 3,
                    [[ 32] * 2] * 4,
                    [[ 64] * 2] * 6,
                    [[128] * 2] * 3]

        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=ResBlock)
        super().__init__(
            OuterBlock(1,                   features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=dense_dropout_p, inplace=True) if dense_dropout_p is not None else None,
            nn.Linear(features[4][-1][-1], n_output))


class ResNet34Large(ResNet):
    def __init__(self, n_output, size, dense_dropout_p=0.5):

        features = [[ [64]],
                    [ [64] * 2] * 3,
                    [[128] * 2] * 4,
                    [[256] * 2] * 6,
                    [[512] * 2] * 3]

        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=ResBlock)
        super().__init__(
            OuterBlock(1,                   features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=dense_dropout_p, inplace=True) if dense_dropout_p is not None else None,
            nn.Linear(features[4][-1][-1], n_output))


class SE3Net(ResNet):
    def __init__(self, res_block, n_output, size,
                 capsule_dropout_p=0.1, dense_dropout_p=0.5,
                 downsample_by_pooling=False):

        features = [[[( 4,  4,  4,  4)] * 1],  # 64 channels
                    [[( 4,  4,  4,  4)] * 1],  # 64 channels
                    [[( 8,  8,  8,  8)] * 1],  # 128 channels
                    [[(16, 16, 16, 16)] * 1],  # 256 channels
                    [[(256,)]]]

        common_params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',  # STILL OPEN TO TEST
            'batch_norm_before_conv': True,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': capsule_dropout_p,  # drop probability of whole capsules
            'downsample_by_pooling': downsample_by_pooling,
        }
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))

        super().__init__(
            OuterBlock((1,),                features[0], size=size),
            OuterBlock(features[0][-1][-1], features[1], size=size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=dense_dropout_p, inplace=True) if dense_dropout_p is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output))

class SE3ResNet34(ResNet):
    def __init__(self, res_block, n_output, size,
                 capsule_dropout_p=0.1, dense_dropout_p=0.5,
                 downsample_by_pooling=False):
        features = [[[( 4,  4,  4,  4)]],          #  64 channels
                    [[( 4,  4,  4,  4)] * 2] * 3,  #  64 channels
                    [[( 8,  8,  8,  8)] * 2] * 4,  # 128 channels
                    [[(16, 16, 16, 16)] * 2] * 6,  # 256 channels
                    [[(32, 32, 32, 32)] * 2] * 2 + [[(32, 32, 32, 32), (512, 0, 0, 0)]]]  # 512 channels
        common_params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',   # STILL OPEN TO TEST
            'batch_norm_before_conv': True,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': capsule_dropout_p, # drop probability of whole capsules
            'downsample_by_pooling': downsample_by_pooling,
        }
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((1,),                features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=dense_dropout_p, inplace=True) if dense_dropout_p is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output))


class SE3ResNet34Large(ResNet):
    def __init__(self, res_block, n_output, size,
                 capsule_dropout_p=0.1, dense_dropout_p=0.5,
                 downsample_by_pooling=False):
        features = [[[( 8,  8,  8,  8)]],          # 128 channels
                    [[( 8,  8,  8,  8)] * 2] * 3,  # 128 channels
                    [[(16, 16, 16, 16)] * 2] * 4,  # 256 channels
                    [[(32, 32, 32, 32)] * 2] * 6,  # 512 channels
                    [[(64, 64, 64, 64)] * 2] * 2 + [[(64, 64, 64, 64), (1024, 0, 0, 0)]]]  # 1024 channels
        common_params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',  # STILL OPEN TO TEST
            'batch_norm_before_conv': True,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': capsule_dropout_p,  # drop probability of whole capsules
            'downsample_by_pooling': downsample_by_pooling,
        }
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((1,),                features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=dense_dropout_p, inplace=True) if dense_dropout_p is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output))


model_classes = {"resnet34_k3":
                     partial(ResNet34, size=3,
                             dense_dropout_p=0.5),
                 "resnet34_large_k3":
                     partial(ResNet34Large, size=3,
                             dense_dropout_p=0.5),
                 "se3net_k3_gated":
                     partial(SE3Net, size=3,
                             dense_dropout_p=None,
                             capsule_dropout_p=None,
                             res_block=SE3GatedResBlock),
                 "se3net_k5_gated":
                     partial(SE3Net, size=5,
                             dense_dropout_p=None,
                             capsule_dropout_p=None,
                             res_block=SE3GatedResBlock),
                 "se3net_k3_norm":
                     partial(SE3Net, size=3,
                             dense_dropout_p=None,
                             capsule_dropout_p=None,
                             res_block=SE3NormResBlock),
                 "se3net_k5_norm":
                     partial(SE3Net, size=5,
                             dense_dropout_p=None,
                             capsule_dropout_p=None,
                             res_block=SE3NormResBlock),
                 "se3resnet34_k3_gated":
                     partial(SE3ResNet34, size=3,
                             dense_dropout_p=0.5,
                             capsule_dropout_p=0.1,
                             res_block=SE3GatedResBlock),
                 "se3resnet34_k5_gated":
                     partial(SE3ResNet34, size=5,
                             dense_dropout_p=0.5,
                             capsule_dropout_p=0.1,
                             res_block=SE3GatedResBlock),
                 "se3resnet34_k3_norm":
                     partial(SE3ResNet34, size=3,
                             dense_dropout_p=0.5,
                             capsule_dropout_p=0.1,
                             res_block=SE3NormResBlock),
                 "se3resnet34_k5_norm":
                     partial(SE3ResNet34, size=5,
                             dense_dropout_p=0.5,
                             capsule_dropout_p=0.1,
                             res_block=SE3NormResBlock),
                 "se3resnet34_large_k3_gated":
                     partial(SE3ResNet34Large, size=3,
                             dense_dropout_p=0.5,
                             capsule_dropout_p=0.1,
                             res_block=SE3GatedResBlock)
                 }


def infer(model, loader):
    model.eval()
    losses = []
    outs = []
    ys = []
    for _, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        y = torch.autograd.Variable(target)
        out = model(x)
        outs.append(out.data.cpu().numpy())
        ys.append(y.data.cpu().numpy())
        losses.append(torch.nn.functional.cross_entropy(out, y, reduce=False).data.cpu().numpy())
    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    return outs, ys, np.concatenate(losses)


def main(args, data_filename, model_class, initial_lr, lr_decay_start, lr_decay_base, batch_size=32, randomize_orientation=False):

    torch.backends.cudnn.benchmark = True

    if args.mode == 'train':
        train_set = torch.utils.data.ConcatDataset([
            Cath(data_filename, split=i, download=True,
                 randomize_orientation=randomize_orientation,
                 discretization_bins=args.data_discretization_bins,
                 discretization_bin_size=args.data_discretization_bin_size) for i in range(7)])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        n_output = len(train_set.datasets[0].label_set)

    if args.mode in ['train', 'validate']:
        validation_set = Cath(
            data_filename, split=7,
            discretization_bins=args.data_discretization_bins,
            discretization_bin_size=args.data_discretization_bin_size)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        n_output = len(validation_set.label_set)

    if args.mode == 'test':
        test_set = torch.utils.data.ConcatDataset([Cath(
            data_filename, split=i,
            discretization_bins=args.data_discretization_bins,
            discretization_bin_size=args.data_discretization_bin_size) for i in range(8, 10)])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        n_output = len(test_set.datasets[0].label_set)

    model = model_class(n_output=n_output)
    if torch.cuda.is_available():
        model.cuda()

    print(model)

    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # split up parameters into groups, named_parameters() returns tupels ('name', parameter)
    # each group gets its own regularization gain
    convLayers = [m for m in model.modules()
                  if isinstance(m, (SE3Convolution, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))]
    normActivs = [m for m in model.modules() if isinstance(m, (NormSoftplus, NormRelu, ScalarActivation))]
    batchnormLayers = [m for m in model.modules() if isinstance(m, (SE3BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    linearLayers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    weights_conv = [p for m in convLayers for n, p in m.named_parameters() if n.endswith('weight')]
    weights_bn = [p for m in batchnormLayers for n, p in m.named_parameters() if n.endswith('weight')]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]  # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    biases_conv = [p for m in convLayers for n, p in m.named_parameters() if n.endswith('bias')]
    biases_activs = [p for m in normActivs for n, p in m.named_parameters() if n.endswith('bias')]
    biases_bn = [p for m in batchnormLayers for n, p in m.named_parameters() if n.endswith('bias')]
    biases_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('bias')]  # CROP OFF LAST WEIGHT !!!!! (classification layer)
    biases_fully, biases_softmax = biases_fully[:-1], [biases_fully[-1]]
    for np_tuple in model.named_parameters():
        if not np_tuple[0].endswith(('weight', 'weights_re', 'weights_im', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(np_tuple[0]))
    param_groups = [dict(params=weights_conv,    lamb_L1=args.lamb_conv_weight_L1,     lamb_L2=args.lamb_conv_weight_L2),
                    dict(params=weights_bn,      lamb_L1=args.lamb_bn_weight_L1,       lamb_L2=args.lamb_bn_weight_L2),
                    dict(params=weights_fully,   lamb_L1=args.lamb_linear_weight_L1,   lamb_L2=args.lamb_linear_weight_L2),
                    dict(params=weights_softmax, lamb_L1=args.lamb_softmax_weight_L1,  lamb_L2=args.lamb_softmax_weight_L2),
                    dict(params=biases_conv,     lamb_L1=args.lamb_conv_bias_L1,       lamb_L2=args.lamb_conv_bias_L2),
                    dict(params=biases_activs,   lamb_L1=args.lamb_norm_activ_bias_L1, lamb_L2=args.lamb_norm_activ_bias_L2),
                    dict(params=biases_bn,       lamb_L1=args.lamb_bn_bias_L1,         lamb_L2=args.lamb_bn_bias_L2),
                    dict(params=biases_fully,    lamb_L1=args.lamb_linear_bias_L1,     lamb_L2=args.lamb_linear_bias_L2),
                    dict(params=biases_softmax,  lamb_L1=args.lamb_softmax_bias_L1,    lamb_L2=args.lamb_softmax_bias_L2)]

    # Check whether all parameters are in groups
    params_in_groups = [id(param) for group in param_groups for param in group['params']]
    if len(list(params_in_groups)) != len(list(model.parameters())):
        error_msg = "The following parameters will not be optimized:\n"
        for name, param in model.named_parameters():
            if id(param) not in params_in_groups:
                error_msg += "\t" + name + "\n"
        raise RuntimeError(error_msg)

    # old version, does not differentiate between parameter groups
    # param_groups = [dict(params=model.parameters(), lamb_L1=lambda_L1,  lamb_L2=lambda_L2)]
    # You can set different regularization for different parameter groups by splitting them up

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = Adam(param_groups, lr=initial_lr)

    # Set up model dumping
    epoch_start_index = 0
    if args.read_from_checkpoint is not None:
        checkpoint_index = args.read_from_checkpoint
        checkpoint_basename = os.path.join(args.model_checkpoint_path,
                                           'model_%s' % (model.__class__.__name__))
        if checkpoint_index == -1:
            import glob
            checkpoint_filename = glob.glob(checkpoint_basename + '_*.ckpt')[-1]
            checkpoint_index = int(checkpoint_filename.split('.')[-2].split('_')[-1])
        else:
            checkpoint_filename = checkpoint_basename+'_%d.ckpt' % checkpoint_index
        print("Restoring model from:", checkpoint_filename)
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        epoch_start_index = checkpoint_index+1

    # Set the logger
    if args.log_to_tensorboard:
        from datetime import datetime
        now = datetime.now()
        logger = Logger('./logs/%s/' % now.strftime("%Y%m%d_%H%M%S"))

    if args.mode == 'train':

        for epoch in range(epoch_start_index, 100):

            # decay learning rate
            optimizer, _ = lr_scheduler_exponential(optimizer, epoch, initial_lr, lr_decay_start, lr_decay_base, verbose=True)
            optimizer.zero_grad()

            training_losses = []
            training_outs = []
            training_accs = []
            for batch_idx, (data, target) in enumerate(train_loader):
                time_start = time.perf_counter()

                target = torch.LongTensor(target)

                model.train()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

                # forward and backward propagation
                out = model(x)
                losses = torch.nn.functional.cross_entropy(out, y, reduce=False)
                loss = losses.mean()
                loss.backward()
                if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
                    optimizer.step()
                    optimizer.zero_grad()

                _, argmax = torch.max(out, 1)
                acc = (argmax.squeeze() == y).float().mean()

                training_losses.append(losses.data.cpu().numpy())
                training_outs.append(out.data.cpu().numpy())
                training_accs.append(acc.data[0])

                print("[{}:{}/{}] loss={:.4} acc={:.2} time={:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    float(loss.data[0]), float(acc.data[0]),
                    time.perf_counter() - time_start))
            loss_avg = np.mean(training_losses)
            acc_avg = np.mean(training_accs)
            training_outs = np.concatenate(training_outs)
            training_losses = np.concatenate(training_losses)

            validation_outs, ys, validation_losses = infer(model, validation_loader)

            # compute the accuracy
            validation_acc = np.sum(validation_outs.argmax(-1) == ys) / len(ys)

            validation_loss_avg = np.mean(validation_losses)

            print('TRAINING SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                loss_avg, acc_avg))
            print('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                validation_loss_avg, validation_acc))

            if args.log_to_tensorboard:

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'training set avg loss': loss_avg,
                    'training set accuracy': acc_avg,
                    'validation set avg loss': validation_loss_avg,
                    'validation set accuracy': validation_acc,
                }

                step = epoch
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step + 1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(),
                                         step + 1)

                # (3) Log losses for all datapoints in validation and training set
                logger.histo_summary("losses/validation/", validation_losses, step+1)
                logger.histo_summary("losses/training", training_losses, step+1)

                # (4) Log losses for all datapoints in validation and training set
                for i in range(n_output):
                    logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step+1)
                    logger.histo_summary("logits/%d/training" % i, training_outs[:, i], step+1)

            if args.save_checkpoints:
                checkpoint_filename = os.path.join(
                    args.model_checkpoint_path,
                    'model_%s_%d.ckpt' % (model.__class__.__name__, epoch))
                torch.save({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           checkpoint_filename)
                print("Model saved to %s" % checkpoint_filename)

    elif args.mode == 'validate':
        out, y, validation_loss_sum = infer(model, validation_loader)

        # compute the accuracy
        validation_acc = np.sum(out.argmax(-1) == y) / len(y)
        validation_loss_avg = validation_loss_sum / len(validation_loader.dataset)

        print('VALIDATION SET: loss={:.4} acc={:.2}'.format(
            validation_loss_avg, validation_acc))

    elif args.mode == 'test':
        out, y, test_loss_sum = infer(model, test_loader)

        # compute the accuracy
        test_acc = np.sum(out.argmax(-1) == y) / len(y)
        test_loss_avg = test_loss_sum / len(test_loader.dataset)

        print('VALIDATION SET: loss={:.4} acc={:.2}'.format(
            test_loss_avg, test_acc))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-filename", choices={"cath_3class.npz", "cath_10arch.npz"}, required=True,
                        help="The name of the data file (will automatically downloaded)")
    parser.add_argument("--data-discretization-bins", type=int, default=50,
                        help="Number of bins used in each dimension for the discretization of the input data")
    parser.add_argument("--data-discretization-bin-size", type=float, default=2.0,
                        help="Size of bins used in each dimension for the discretization of the input data")
    parser.add_argument("--model", choices=model_classes.keys(), required=True,
                        help="Which model definition to use")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--log-to-tensorboard", action="store_true", default=False,
                        help="Whether to output log information in tensorboard format (default: %(default)s)")
    parser.add_argument("--model-checkpoint-path", type=str, default="models",
                        help="Where to dump/read model checkpoints (default: %(default)s)")
    parser.add_argument("--save-checkpoints", action="store_true", default=False,
                        help="Save model checkpoints at each epoch")
    parser.add_argument("--read-from-checkpoint", type=int, default=None,
                        help="Read model from checkpoint given by index")
    parser.add_argument("--mode", choices=['train', 'test', 'validate'], default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--initial_lr", default=1e-3, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=1,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=1,
                        help="exponential decay factor per epoch")
    # parser.add_argument("--lambda_L1", default=0, type=float,
    #                     help="L1 regularization factor")
    # parser.add_argument("--lambda_L2", default=0, type=float,
    #                     help="L2 regularization factor")

    # WEIGHTS
    parser.add_argument("--lamb_conv_weight_L1", default=0, type=float,
                        help="L1 regularization factor for convolution weights")
    parser.add_argument("--lamb_conv_weight_L2", default=0, type=float,
                        help="L2 regularization factor for convolution weights")
    parser.add_argument("--lamb_bn_weight_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_bn_weight_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_linear_weight_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_linear_weight_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_softmax_weight_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer weights")
    parser.add_argument("--lamb_softmax_weight_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer weights")
    # BIASES
    parser.add_argument("--lamb_conv_bias_L1", default=0, type=float,
                        help="L1 regularization factor for convolution biases")
    parser.add_argument("--lamb_conv_bias_L2", default=0, type=float,
                        help="L2 regularization factor for convolution biases")
    parser.add_argument("--lamb_norm_activ_bias_L1", default=0, type=float,
                        help="L1 regularization factor for norm activation biases")
    parser.add_argument("-lamb_norm_activ_bias_L2", default=0, type=float,
                        help="L2 regularization factor for norm activation biases")
    parser.add_argument("--lamb_bn_bias_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_bn_bias_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_linear_bias_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_linear_bias_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_softmax_bias_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer biases")
    parser.add_argument("--lamb_softmax_bias_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer biases")

    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    if not os.path.exists(args.model_checkpoint_path):
        os.mkdir(args.model_checkpoint_path)

    main(
        args=args,
        data_filename=args.data_filename,
        model_class=model_classes[args.model],
        initial_lr=args.initial_lr,
        lr_decay_start=args.lr_decay_start,
        lr_decay_base=args.lr_decay_base,
        batch_size=args.batch_size,
        randomize_orientation=args.randomize_orientation,
    )
