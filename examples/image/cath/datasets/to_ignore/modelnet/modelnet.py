# pylint: disable=E1101,R,C
import glob
import os
import numpy as np
import torch
import torch.utils.data
import shutil
from functools import partial

from scipy.ndimage import affine_transform
from e3nn.SO3 import rot



def get_modelnet_loader(root_dir, dataset, mode, size, data_loader_kwargs, args):
    if dataset == 'ModelNet10':
        classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    else:
        raise NotImplementedError('Other datasets than ModelNet10 not fully implemented yet')
    def _get_trafo(size, args):
        trafos = []
        if args.add_z_axis:
            trafos.append(AddZAxis(zmin=-size/2, zmax=size/2))
        affine_trafo_args = {'scale': (1,args.augment_scales) if args.augment_scales is not False else False,
                             'flip': args.augment_flip,
                             'translate': args.augment_translate,
                             'rotate': args.augment_rotate}
        if not all(False for val in affine_trafo_args.values()):
            trafos.append(RandomAffine3d(vol_shape=(size,size,size), **affine_trafo_args))
        if len(trafos) == 0:
            return None
        else:
            from torchvision.transforms import Compose
            return Compose(trafos)
    transform = _get_trafo(size, args)
    dataset = ModelNet(root_dir, dataset, mode, size, classes, transform)
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_kwargs)
    return dataset, data_loader


class ModelNet(torch.utils.data.Dataset):
    ''' '''
    def __init__(self, root_dir, dataset, mode, size, classes, transform=None, target_transform=None):
        '''
        :param root: directory to store dataset in
        :param dataset:
        :param mode: dataset to load: 'train', 'validation', 'test' or 'train_full'
                     the validation set is split from the train set, the full train set can be accessed via 'train_full'
                    :param transform: transformation applied to image in __getitem__
                                      currently used to load cached file from string
                    :param target_transform: transformation applied to target in __getitem__
        '''
        self.root = os.path.expanduser(root_dir)
        assert dataset in ['ModelNet10', 'ModelNet40']
        self.dataset = dataset

        assert mode in ['train', 'validation', 'test', 'train_full']
        self.mode = mode
        self.size = size
        self.classes = classes

        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train_full':
            self.file_names =  sorted(glob.glob(os.path.join(self.root, self.dataset, '*', 'train',      '*_size{}.npy'.format(self.size))))
            self.file_names += sorted(glob.glob(os.path.join(self.root, self.dataset, '*', 'validation', '*_size{}.npy'.format(self.size))))
        else:
            self.file_names =  sorted(glob.glob(os.path.join(self.root, self.dataset, '*', self.mode,    '*_size{}.npy'.format(self.size))))
        assert self.__len__() > 0
        print('Loaded dataset \'{}\', size \'{}\' in mode \'{}\' with {} elements'.format(self.dataset, self.size, self.mode, self.__len__()))

    def __getitem__(self, index):
        img_fname = self.file_names[index]

        img = np.load(img_fname).astype(np.int8).reshape((1, self.size, self.size, self.size))

        target_class_string = img_fname.split(os.path.sep)[-3]
        target = self.classes.index(target_class_string)

        if self.transform is not None:
            img = self.transform(img)
            img = torch.from_numpy(img.astype(np.float32))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.file_names)


class AddZAxis(object):
    ''' add z-axis as second channel to volume
        the scale of the z-axis can be set freely
        if the volume tensor does not contain a channel dimension, add it
        the z axis is assumed to be the last axis of the volume tensor
    '''
    def __init__(self, zmin, zmax):
        ''' :param zmin: min z-value
            :param zmin: max z-value
        '''
        self.zmin = zmin
        self.zmax = zmax

    def __call__(self, sample):
        assert sample.ndim in (3,4)
        if sample.ndim == 3:
            sample = sample[np.newaxis,...]
        broadcast_shape = list(sample.shape)
        broadcast_shape[0] = 1
        zsize = sample.shape[-1]
        zcoords = np.linspace(self.zmin, self.zmax, num=zsize, endpoint=True)
        zcoords = np.broadcast_to(zcoords, broadcast_shape)
        return np.concatenate([sample,zcoords], axis=0)


class RandomAffine3d(object):
    ''' random affine transformation applied to volume center
        assumes volume with channel dimension, shape (C,X,Y,Z)
    '''
    def __init__(self, vol_shape, scale=(.9,1.1), flip=True, translate=True, rotate=True):
        ''' :param vol_shape: shape of the volumes (X,Y,Z), needed to compute center
            :param scale: False or tuple giving min and max scale value
                          VALUES <1 ZOOM IN !!!
            :param flip: bool controlling random reflection with p=0.5
            :param trans: bool controlling uniform random translations in (-.5, .5) on all axes
            :param rotate: bool controlling uniform random rotations
        '''
        self.vol_shape = np.array(vol_shape)
        self.scale = scale if scale is not False else (1,1)
        self.flip = flip
        self.translate = translate
        self.rotate = rotate

    def __call__(self, sample):
        assert sample.ndim == 4
        trafo = self._get_random_affine_trafo()
        return np.stack([trafo(channel) for channel in sample])

    def _get_random_affine_trafo(self):
        if self.rotate:
            alpha,beta,gamma = np.pi*np.array([2,1,2])*np.random.rand(3)
            aff = rot(alpha,beta,gamma)
        else:
            aff = np.eye(3) # only non-homogeneous coord part
        fl = (-1)**np.random.randint(low=0, high=2) if self.flip else 1
        if self.scale is not None:
            sx,sy,sz = np.random.uniform(low=self.scale[0], high=self.scale[1], size=3)
        else:
            sx,sy,sz = 1
        aff[:,0] *= sx*fl
        aff[:,1] *= sy
        aff[:,2] *= sz
        center = self.vol_shape/2
        offset = center - center@aff.T # correct offset to apply trafo around center
        if self.translate:
            offset += np.random.uniform(low=-.5, high=.5, size=3)
        return partial(affine_transform, matrix=aff, offset=offset)
