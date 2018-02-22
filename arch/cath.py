# pylint: disable=C,R,E1101
'''
Architecture to predict the structural categories of proteins according to the CATH 
classification (www.cathdb.info).

'''
import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import math
import scipy.io
import os
import time
from timeit import default_timer as timer
from scipy.stats import special_ortho_group

from se3_cnn.blocks import GatedBlock
from se3_cnn.blocks import NormBlock
from se3_cnn.batchnorm import SE3BatchNorm
from se3_cnn.convolution import SE3Convolution
from se3_cnn import basis_kernels

from se3_cnn.non_linearities import NormRelu
from se3_cnn.non_linearities import NormSoftplus

from se3_cnn.util.optimizers_L1L2 import Adam
from se3_cnn.util.lr_schedulers import lr_scheduler_exponential

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


class Cath(torch.utils.data.Dataset):
    url = 'https://github.com/deepfold/cath_datasets/blob/master/{}?raw=true'

    def __init__(self, dataset, split, download=False, use_density=True, randomize_orientation=False):
        self.root = os.path.expanduser("cath")

        if download:
            self.download(dataset)

        self.use_density = use_density
        self.randomize_orientation = randomize_orientation
        
        if not self._check_exists(dataset):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data = np.load(os.path.join(self.root, dataset))
        split_start_indices = data['split_start_indices']
        split_range = list(zip(split_start_indices[0:], list(split_start_indices[1:])+[None]))[split]
        self.positions = data['positions'][split_range[0]:split_range[1]]
        self.atom_types = data['atom_types'][split_range[0]:split_range[1]]
        self.n_atoms = data['n_atoms'][split_range[0]:split_range[1]]
        self.labels = [tuple(v) if len(v)>1 else v[0] for v in data['labels'][split_range[0]:split_range[1]]]

        self.atom_type_set = np.unique(self.atom_types[0][:self.n_atoms[0]])
        self.n_atom_types = len(self.atom_type_set)
        self.atom_type_map = dict(zip(self.atom_type_set, range(len(self.atom_type_set))))

        self.label_set = sorted(list(set(self.labels)))
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))
        
    def __getitem__(self, index):

        time_stamp = timer()
        
        n_atoms    = self.n_atoms[index]
        positions  = self.positions[index][:n_atoms]
        atom_types = self.atom_types[index][:n_atoms]
        label      = self.label_map[self.labels[index]]

        p = 2.0
        n = 50

        if torch.cuda.is_available():
            fields = torch.cuda.FloatTensor(*(self.n_atom_types,)+(n,n,n)).fill_(0)
        else:
            fields = torch.zeros(*(self.n_atom_types,)+(n,n,n))

        if self.randomize_orientation:
            random_rotation = special_ortho_group.rvs(3)
            positions = np.dot(random_rotation,positions.T).T

        if self.use_density:
        
            ## Numpy version ##
            # a = np.linspace(start=-n / 2 * p + p / 2, stop=n / 2 * p - p / 2, num=n, endpoint=True)
            # xx, yy, zz = np.meshgrid(a, a, a, indexing="ij")

            # fields_np = np.zeros((self.n_atom_types, n, n, n), dtype=np.float32)
            # for i, atom_type in enumerate(self.atom_type_set):

            #     # Extract positions with current atom type
            #     pos = positions[atom_types == atom_type]

            #     # Create grid x atom_pos grid
            #     posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
            #     posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
            #     posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))                            

            #     # Calculate density
            #     density = np.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (p)**2))

            #     # Normalize so each atom density sums to one
            #     density /= np.sum(density, axis=0)

            #     # Sum densities and reshape to original shape
            #     fields_np[i] = np.sum(density, axis=1).reshape(xx.shape)


            ## Pytorch version ##        

            # Create linearly spaced grid
            a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)
            if torch.cuda.is_available():
                a = a.cuda()

            # Pytorch does not suppoert meshgrid - do the repeats manually
            xx = a.view(-1,1,1).repeat(1, len(a), len(a))
            yy = a.view(1,-1,1).repeat(len(a), 1, len(a))
            zz = a.view(1,1,-1).repeat(len(a), len(a), 1)

            for i, atom_type in enumerate(self.atom_type_set):

                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Transfer position vector to gpu
                pos = torch.FloatTensor(pos)
                if torch.cuda.is_available():
                    pos = pos.cuda()

                # Pytorch does not suppoert meshgrid - do the repeats manually
                # Numpy equivalent:
                # posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
                # posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
                # posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))                            
                xx_xx = xx.view(-1, 1).repeat(1, len(pos))
                posx_posx = pos[:,0].contiguous().view(1, -1).repeat(len(xx.view(-1)), 1)
                yy_yy = yy.view(-1, 1).repeat(1, len(pos))
                posy_posy = pos[:,1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
                zz_zz = zz.view(-1, 1).repeat(1, len(pos))
                posz_posz = pos[:,2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

                # Calculate density
                sigma = 0.5*p
                density = torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

                # Normalize so each atom density sums to one
                density /= torch.sum(density, dim=0)

                # Sum densities and reshape to original shape
                fields[i] = torch.sum(density, dim=1).view(xx.shape)
        else:

            for i, atom_type in enumerate(self.atom_type_set):
            
                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Lookup indices and move to GPU
                indices = torch.LongTensor(np.ravel_multi_index(np.digitize(pos, a+p/2).T, dims=(n,n,n)))
                if torch.cuda.is_available():
                    indices = indices.cuda()

                # Set values
                fields[i].view(-1)[indices] = 1

            
        # assert((np.abs(fields.numpy() - fields_np)<0.001).all())


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), s=10*fields.numpy().reshape(-1), c=fields.numpy().reshape(-1), cmap=plt.get_cmap("Blues"))
        # plt.show()
        # plt.savefig("grid.png")

        time_elapsed = timer() - time_stamp
        # print("Time spent on __getitem__: %.4f sec" % time_elapsed)
        
        return fields, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self, dataset):
        return os.path.exists(os.path.join(self.root, dataset))

    def download(self, dataset):
        from six.moves import urllib

        if self._check_exists(dataset):
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url.format(dataset))
        data = urllib.request.urlopen(self.url.format(dataset))
        file_path = os.path.join(self.root, dataset)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Done!')


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)

class SE3Net(nn.Module):

    def __init__(self, n_output):
        super().__init__()

        # The parameters of a GatedBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1, ),  # As input we have a scalar field
            (1,  3,  5, 5),  # 70 channels
            (1,  3,  5, 5),  # 70 channels
            (2,  6, 10, 10),  # 140 channels
            (4, 12, 20, 20),  # 280 channels
            (140, )
        ]

        from se3_cnn import basis_kernels
        radial_window_dict = {
            'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
            # 'radial_window_fct_kwargs': {'mode': 'sfcnn', 'border_dist': 0.,
            'radial_window_fct_kwargs': {'mode': 'compromise', 'border_dist': 0., 'sigma': .6}}
        common_block_params = {'size': 7, 'padding': 3,
                               'batch_norm_momentum': 0.01,
                               'batch_norm_mode': 'maximum',
                               'radial_window_dict': radial_window_dict,
                               'batch_norm_before_conv': False}

        block_params = [
            {'activation': torch.nn.functional.softplus, 'stride': 1},
            {'activation': torch.nn.functional.softplus, 'stride': 2},
            {'activation': torch.nn.functional.softplus, 'stride': 2},
            {'activation': torch.nn.functional.softplus, 'stride': 2},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1],
                             **common_block_params, **block_params[i])
                  for i in range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(140, n_output),
            # torch.nn.ReLU(),
            # torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.sequence(x)


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, size=3, stride=1):
        super().__init__()

        channels = [channels_in] + channels_out

        self.layers = []
        for i in range(len(channels) - 1):
            self.layers += [
                nn.Conv3d(channels[i], channels[i+1],
                          kernel_size=size,
                          padding=size//2,
                          stride=stride if i==0 else 1,
                          bias=False),
                nn.BatchNorm3d(channels[i+1]),
                nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        # Add shortcut if number of layers is larger than 1
        if len(channels_out) > 1:
            self.shortcut = nn.Sequential(*[
                nn.Conv3d(channels[0], channels[-1],
                          kernel_size=1,
                          padding=0,
                          stride=stride,
                          bias=False),
                nn.BatchNorm3d(channels[-1])])

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
        return out



class SE3ResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 norm_activation=True,
                 radial_window_dict=None,
                 batch_norm_momentum=0.01,
                 batch_norm_mode='maximum',
                 batch_norm_before_conv=False,
                 activation=torch.nn.functional.relu,
                 **kwargs):
        super().__init__()

        reprs = [in_repr] + out_reprs

        if radial_window_dict is None:
            radial_window_dict = {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}}

        self.layers = []
        for i in range(len(reprs) - 1):
            self.layers.append(
                (GatedBlock if not norm_activation else NormBlock)(
                    reprs[i], reprs[i + 1], size=size, padding=size//2,
                    stride=stride if i == 0 else 1,
                    radial_window_dict=radial_window_dict,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_mode=batch_norm_mode,
                    batch_norm_before_conv=batch_norm_before_conv,
                    activation=activation,
                    **kwargs))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        # Add shortcut if number of layers is larger than 1
        if len(out_reprs) > 1:
            self.shortcut = (GatedBlock if not norm_activation else NormBlock)(
                reprs[0], reprs[-1], size=1, padding=0, stride=stride,
                radial_window_dict=radial_window_dict,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_mode=batch_norm_mode,
                batch_norm_before_conv=batch_norm_before_conv,
                activation=None,
                **kwargs)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNet34(ResNet):
    def __init__(self, n_output):
        features = [ [16] * 3,
                     [32] * 4,
                     [64] * 6,
                    [128] * 3]
        super().__init__(
            ResBlock(1,              features[0], size=3),
            ResBlock(features[0][0], features[1], size=3, stride=2),
            ResBlock(features[1][0], features[2], size=3, stride=2),
            ResBlock(features[2][0], features[3], size=3, stride=2),
            AvgSpacial(),
            nn.Linear(features[3][-1], n_output))


class SE3Net_k5(ResNet):
    def __init__(self, n_output):
        features = [[(4, 4, 4)] * 1,
                    [(4, 4, 4)] * 1,
                    [(8, 8, 8)] * 1,
                    [(8, 8, 8)] * 1,
                    [(128,)]]
        params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',
            'batch_norm_before_conv': False,
            'activation': torch.nn.functional.relu,
            'norm_activation': True,
            'activation_bias_min': 0.5,
            'activation_bias_max': 2.0,
        }
        super().__init__(
            SE3ResBlock((1,),           features[0], size=5, **params),
            SE3ResBlock(features[0][0], features[1], size=5, stride=2, **params),
            SE3ResBlock(features[1][0], features[2], size=5, stride=2, **params),
            SE3ResBlock(features[2][0], features[3], size=5, stride=2, **params),
            SE3ResBlock(features[3][0], features[4], size=3, stride=1, **params),
            AvgSpacial(),
            nn.Linear(features[4][-1][0], n_output))


class SE3Net_k7(ResNet):
    def __init__(self, n_output):
        features = [[(4, 4, 4)] * 1,
                    [(4, 4, 4)] * 1,
                    [(8, 8, 8)] * 1,
                    [(8, 8, 8)] * 1,
                    [(128,)]]
        params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',
            'batch_norm_before_conv': False,
            'activation': torch.nn.functional.relu,
            'norm_activation': True,
            'activation_bias_min': 0.5,
            'activation_bias_max': 2.0,
        }
        super().__init__(
            SE3ResBlock((1,),           features[0], size=7, **params),
            SE3ResBlock(features[0][0], features[1], size=7, stride=2, **params),
            SE3ResBlock(features[1][0], features[2], size=7, stride=2, **params),
            SE3ResBlock(features[2][0], features[3], size=7, stride=2, **params),
            SE3ResBlock(features[3][0], features[4], size=3, stride=1, **params),
            AvgSpacial(),
            nn.Linear(features[4][-1][0], n_output))

class SE3ResNet34(ResNet):
    def __init__(self, n_output):
        features = [[(2,  6, 10)] * 3,
                    [(2,  6, 10)] * 4,
                    [(4, 12, 20)] * 6,
                    [(8, 24, 40)] * 2 + [(8*1 + 24*3 + 40*5, 0, 0)]]
        params = {
            'radial_window_dict': {
                'radial_window_fct': basis_kernels.gaussian_window_fct_convenience_wrapper,
                'radial_window_fct_kwargs': {'mode': 'compromise',
                                             'border_dist': 0.,
                                             'sigma': .6}},
            'batch_norm_momentum': 0.01,
            'batch_norm_mode': 'maximum',
            'batch_norm_before_conv': False,
            'activation': torch.nn.functional.relu,
            'norm_activation': True,
            'activation_bias_min': 0.5,
            'activation_bias_max': 2.0,
        }
        super().__init__(
            SE3ResBlock((1,),           features[0], size=5, **params),
            SE3ResBlock(features[0][0], features[1], size=5, stride=2, **params),
            SE3ResBlock(features[1][0], features[2], size=5, stride=2, **params),
            SE3ResBlock(features[2][0], features[3], size=5, stride=2, **params),
            AvgSpacial(),
            nn.Linear(features[3][-1][0], n_output))


model_classes = {"se3net": SE3Net,
                 "resnet34": ResNet34,
                 "se3resnet34": SE3ResNet34,
                 "se3net_k5": SE3Net_k5,
                 "se3net_k7": SE3Net_k7}

def main(args, data_filename, model_class, initial_lr, lr_decay_start, lr_decay_base, batch_size=32, randomize_orientation=False):

    torch.backends.cudnn.benchmark = True

    train_set = torch.utils.data.ConcatDataset([Cath(data_filename, split=i, download=True, randomize_orientation=randomize_orientation) for i in range(7)])
    validation_set = Cath(data_filename, split=7)
    test_set = torch.utils.data.ConcatDataset([Cath(data_filename, split=i) for i in range(8,10)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    n_output = len(validation_set.label_set)
    
    model = model_class(n_output = n_output)
    if torch.cuda.is_available():
        model.cuda()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


    # split up parameters into groups, named_parameters() returns tupels ('name', parameter)
    # each group gets its own regularization gain
    convLayers      = [m for m in model.modules() if isinstance(m, (SE3Convolution, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))]
    normActivs      = [m for m in model.modules() if isinstance(m, (NormSoftplus, NormRelu))]
    batchnormLayers = [m for m in model.modules() if isinstance(m, (SE3BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    linearLayers    = [m for m in model.modules() if isinstance(m, nn.Linear)]
    weights_conv  = [p for m in convLayers      for n,p in m.named_parameters() if n.endswith('weight')]
    weights_bn    = [p for m in batchnormLayers for n,p in m.named_parameters() if n.endswith('weight')]
    weights_fully = [p for m in linearLayers    for n,p in m.named_parameters() if n.endswith('weight')] # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    biases_conv   = [p for m in convLayers      for n,p in m.named_parameters() if n.endswith('bias')]
    biases_activs = [p for m in normActivs      for n,p in m.named_parameters() if n.endswith('bias')]
    biases_bn     = [p for m in batchnormLayers for n,p in m.named_parameters() if n.endswith('bias')]
    biases_fully  = [p for m in linearLayers    for n,p in m.named_parameters() if n.endswith('bias')] # CROP OFF LAST WEIGHT !!!!! (classification layer)
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
    # old version, does not differentiate between parameter groups
    # param_groups = [dict(params=model.parameters(), lamb_L1=lambda_L1,  lamb_L2=lambda_L2)] # You can set different regularization for different parameter groups by splitting them up

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = Adam(param_groups, lr=initial_lr)

    # Set the logger
    if tensorflow_available:
        logger = Logger('./logs')

    for epoch in range(100):

        # decay learning rate
        optimizer, _ = lr_scheduler_exponential(optimizer, epoch, initial_lr, lr_decay_start, lr_decay_base, verbose=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            time_start = time.perf_counter()

            target = torch.LongTensor(target)
            
            model.train()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

            # forward and backward propagation
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(out, 1)
            acc = (argmax.squeeze() == y).float().mean()

            print("[{}:{}/{}] loss={:.4} acc={:.2} time={:.2}".format(
                epoch, batch_idx, len(train_loader),
                float(loss.data[0]), float(acc.data[0]),
                time.perf_counter() - time_start))

            if tensorflow_available:
                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': loss.data[0],
                    'accuracy': acc.data[0]
                }

                step = epoch*len(train_loader)+batch_idx
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step+1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)


        model.eval()
        loss_sum = 0
        outs = []
        ys = []
        for batch_idx, (data, target) in enumerate(validation_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            x, y = torch.autograd.Variable(data, volatile=True), torch.autograd.Variable(target)
            out = model(x)
            outs.append(out.data.cpu().numpy())
            ys.append(y.data.cpu().numpy())
            loss_sum += torch.nn.functional.cross_entropy(out, y, size_average=False).data[0]  # sum up batch loss
            # print("{}/{}".format(batch_idx, len(validation_loader)))

        out = np.concatenate(outs)
        y = np.concatenate(ys)

        # compute the accuracy
        acc = np.sum(out.argmax(-1) == y) / len(y)

        avg_loss = loss_sum / len(validation_loader.dataset)

        print('VALIDATION [{}:{}/{}] loss={:.4} acc={:.2}'.format(epoch, len(train_loader)-1, len(train_loader), avg_loss, acc))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-filename",
                        help="The name of the data file (will automatically downloaded)")
    parser.add_argument("--model", choices=model_classes.keys(), default='CNN',
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Size of mini batches to use (default: %(default)s)")
    parser.add_argument("--initial_lr", default=1e-3, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float,
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

    main(args=args,
         data_filename=args.data_filename,
         model_class=model_classes[args.model],
         initial_lr=args.initial_lr,
         lr_decay_start=args.lr_decay_start,
         lr_decay_base=args.lr_decay_base,
         batch_size=args.batch_size,
         randomize_orientation=args.randomize_orientation,
         )
