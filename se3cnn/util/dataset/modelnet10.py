# pylint: disable=E1101,R,C
"""
typical usage

https://github.com/antigol/obj2voxel is needed

from se3cnn.util.dataset.modelnet10 import CacheNPY, Obj2Voxel, ModelNet10

cache = CacheNPY("v64", transform=Obj2Voxel(64))

def transform(x):
    x = cache(x)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

def target_transform(x):
    classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    return classes.index(x)

dataset = ModelNet10("./modelnet10/", "train", download=True, transform=transform, target_transform=target_transform)
"""

import glob
import os
import numpy as np
import torch
import torch.utils.data
import subprocess
import random
import sys


class Obj2Voxel:
    def __init__(self, size, rotate=False, zrotate=False, double=False, diagonal_bounding_box=False, diagonal_bounding_box_xy=False):
        self.size = size
        self.rotate = rotate
        self.zrotate = zrotate
        self.double = double
        self.diagonal_bounding_box = diagonal_bounding_box
        self.diagonal_bounding_box_xy = diagonal_bounding_box_xy

    def __call__(self, file_path):
        tmpfile = '%030x.npy' % random.randrange(16**30)
        command = ["obj2voxel", "--size", str(self.size), file_path, tmpfile]
        if self.rotate:
            command += ["--rotate"]
        if self.zrotate:
            command += ["--alpha_rot", str(np.random.rand() * 2 * np.pi)]
        if self.double:
            command += ["--double"]
        if self.diagonal_bounding_box:
            command += ["--diagonal_bounding_box"]
        if self.diagonal_bounding_box_xy:
            command += ["--diagonal_bounding_box_xy"]
        subprocess.run(command)
        x = np.load(tmpfile).astype(np.int8).reshape((self.size, self.size, self.size))
        os.remove(tmpfile)
        return x


class CacheNPY:
    def __init__(self, prefix, transform, repeat=1, pick_randomly=True):
        ''' data loading from .obj, randomized voxelization and caching in .npy files
            :param prefix: cache filename prefix
            :param transform: callable, the transformation applied
                              accepts the file_path as string, returns np.array
            :param repeat: number of transformed instantiations
            :param pick_randomly: whether to pick a random instantiation or return all instantiations in a list
        '''
        self.prefix = prefix
        self.transform = transform
        self.repeat = repeat
        self.pick_randomly = pick_randomly

    def check_trans(self, file_path):
        ''' try to apply transform and return transformed image, optionally raise exception
            meant to voxelize .obj data via Obj2Voxel which consumes a filename
        '''
        print("transform {}...".format(file_path))
        try:
            return self.transform(file_path)
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def __call__(self, file_path):
        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, self.prefix + root + '_{0}.npy')

        exists = [os.path.exists(npy_path.format(i)) for i in range(self.repeat)]

        # if all exist, return random instantiation
        if self.pick_randomly and all(exists):
            i = np.random.randint(self.repeat)
            try:
                return np.load(npy_path.format(i))
            except OSError:
                exists[i] = False

        # if not all exist, create next instantiation and return it
        if self.pick_randomly:
            img = self.check_trans(file_path)
            # store in first non-existing npy_path
            np.save(npy_path.format(exists.index(False)), img)
            return img

        # if not self.pick_randomly return list of all transforms
        output = []
        for i in range(self.repeat):
            try:
                img = np.load(npy_path.format(i))
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(prefix={0}, transform={1})'.format(self.prefix, self.transform)


class ModelNet10(torch.utils.data.Dataset):
    ''' Download ModelNet and output valid obj files content '''

    url_data = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
    # url_data40 = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'

    def __init__(self, root, mode, download=False, transform=None, target_transform=None):
        '''
        :param root: directory to store dataset in
        :param mode: dataset to load: 'train' or 'test'
        :param download: whether on not to download data
        :param transform: transformation applied to image in __getitem__
        :param target_transform: transformation applied to target in __getitem__
        '''
        self.root = os.path.expanduser(root)

        assert mode in ['train', 'test']
        self.mode = mode

        self.transform = transform
        self.target_transform = target_transform

        if download and not self._check_exists():
            self.download_and_process()

        if not self._check_exists():
            print('Dataset not found. You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.root, "ModelNet10", "*", self.mode, "*.obj")))

    def __getitem__(self, index):
        img = self.files[index]  # FILENAME of the image
        target = img.split(os.path.sep)[-3]  # NAME of the class

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.files)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.root, "ModelNet10", "*", "*", "*.obj"))

        return len(files) == 4899

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

        if os.path.exists(file_path):
            return file_path

        print('Downloading ' + url)

        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
                print(".", end="")
                sys.stdout.flush()

        return file_path

    def _unzip(self, file_path):
        import zipfile

        files = glob.glob(os.path.join(self.root, "ModelNet10", "*", "*", "*.off"))
        if len(files) == 4899:
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _off2obj(self):
        files = glob.glob(os.path.join(self.root, "ModelNet10", "*", "*", "*.off"))
        for file_name in sorted(files):
            output_file = file_name.replace(".off", ".obj")
            if os.path.exists(output_file):
                continue

            print("Convert {} into {}".format(file_name, output_file))
            sys.stdout.flush()

            with open(file_name, "rt") as fi:
                data = fi.read().split("\n")

            assert data[0] == "OFF"
            n, m, _ = [int(x) for x in data[1].split()]
            vertices = data[2: 2 + n]
            faces = [x.split()[1:] for x in data[2 + n: 2 + n + m]]
            result = "o object\n"
            for v in vertices:
                result += "v " + v + "\n"

            for f in faces:
                result += "f " + " ".join(str(int(x) + 1) for x in f) + "\n"

            with open(output_file, "wt") as fi:
                fi.write(result)

    def download_and_process(self):

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        zipfile_path = self._download(self.url_data)
        self._unzip(zipfile_path)
        self._off2obj()
