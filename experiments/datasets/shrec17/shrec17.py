# pylint: disable=E1101,R,C
import glob
import os
import re
import csv
import numpy as np
import torch
import torch.utils.data
import subprocess

"""
typical usage

https://github.com/antigol/obj2voxel is needed

transform = Obj2Voxel(64)
transform = CacheNPY("v64", repeat=5, transform=transform)

def target_transform(x):
    classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                '04401088', '04460130', '04468005', '04530566', '04554684']
    return classes.index(x[0])

dataset = Shrec17("./shrec17/", "train", perturbed=True, download=True, transform=transform, target_transform=target_transform)
"""


class Obj2Voxel:
    def __init__(self, size, tmpfile="tmp.npy"):
        self.size = size
        self.tmpfile = tmpfile

    def __call__(self, file_path):
        subprocess.run(["obj2voxel", "--size", str(self.size), file_path, self.tmpfile])
        return np.load(self.tmpfile).astype(np.int8).reshape((self.size, self.size, self.size))


class CacheNPY:
    def __init__(self, prefix, repeat, transform, pick_randomly=True):
        self.transform = transform
        self.prefix = prefix
        self.repeat = repeat
        self.pick_randomly = pick_randomly

    def check_trans(self, file_path):
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

        if self.pick_randomly and all(exists):
            i = np.random.randint(self.repeat)
            try:
                return np.load(npy_path.format(i))
            except OSError:
                exists[i] = False

        if self.pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

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


class Shrec17(torch.utils.data.Dataset):
    '''
    Download SHREC17 and output valid obj files content
    '''

    url_data = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.zip'
    url_label = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.csv'

    def __init__(self, root, dataset, perturbed=True, download=False, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)

        if not dataset in ["train", "test", "val"]:
            raise ValueError("Invalid dataset")

        self.dir = os.path.join(self.root, dataset + ("_perturbed" if perturbed else ""))
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download(dataset, perturbed)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*.obj')))
        if dataset != "test":
            with open(os.path.join(self.root, dataset + ".csv"), 'rt') as f:
                reader = csv.reader(f)
                self.labels = {}
                for row in [x for x in reader][1:]:
                    self.labels[row[0]] = (row[1], row[2])
        else:
            self.labels = None

    def __getitem__(self, index):
        img = f = self.files[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            i = os.path.splitext(os.path.basename(f))[0]
            target = self.labels[i]

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        else:
            return img

    def __len__(self):
        return len(self.files)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.obj"))

        return len(files) > 0

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

        return file_path

    def _unzip(self, file_path):
        import zipfile

        if os.path.exists(self.dir):
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _fix(self):
        print("Fix obj files")

        r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

        path = os.path.join(self.dir, "*.obj")
        files = sorted(glob.glob(path))

        c = 0
        for i, f in enumerate(files):
            with open(f, "rt") as x:
                y = x.read()
                yy = r.sub(r"f \1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
            print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

    def download(self, dataset, perturbed):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        url = self.url_data.format(dataset + ("_perturbed" if perturbed else ""))
        file_path = self._download(url)
        self._unzip(file_path)
        self._fix()

        if dataset != "test":
            url = self.url_label.format(dataset)
            self._download(url)

        print('Done!')
