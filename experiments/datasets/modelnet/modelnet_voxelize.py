# pylint: disable=E1101,R,C
import glob
import os
import numpy as np
import shutil
import argparse
import subprocess
from joblib import Parallel, delayed


def generate_obj_dataset(url, root_dir, classes, N_validation_split, dataset):
    ''' download and convert modelnet10 dataset to .obj, split off validation set
        :param url: url from which to download the dataset
        :param root_dir: the root directory in which the dataset is stored
        :param classes: list of strings with the classes in the dataset
        :param N_validation_split: number of samples per class split off for validation
    '''
    def _download(url, root_dir):
        filename = url.split('/')[-1]
        file_path = os.path.join(root_dir, filename)
        if not os.path.exists(file_path):
            print('Downloading ' + url)
            import requests
            r = requests.get(url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
        return file_path

    def _unzip(file_path, root_dir, dataset):
        if not os.path.exists(os.path.join(root_dir, dataset)):
            import zipfile
            print('Unzip ' + file_path)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(root_dir)
            zip_ref.close()
            os.unlink(file_path)

    def _off2obj(root_dir, dataset):
        print('Convert OFF into OBJ')
        files = glob.glob(os.path.join(root_dir, dataset, "*", "*", "*.off"))
        for file_name in files:
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
            with open(file_name.replace(".off", ".obj"), "wt") as fi:
                fi.write(result)

    def _split_validation(classes, N_valid, dataset):
        ''' split full training set in a training set and a validation set '''
        for cl in classes:
            fnames_cl = sorted(glob.glob(os.path.join(root_dir, dataset, cl, 'train', cl+"*")))
            assert len(fnames_cl) > 0, 'no samples found for class \'{}\''.format(cl)
            valid_dir = os.path.join(root_dir, dataset, cl, 'validation')
            if not os.path.exists(valid_dir):
                os.mkdir(valid_dir)
            for path_train in fnames_cl[-N_valid:]:
                path_valid = path_train.replace('train', 'validation')
                shutil.move(src=path_train, dst=path_valid)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    zipfile_path = _download(url_data, root_dir)
    _unzip(zipfile_path, root_dir, dataset)
    _off2obj(root_dir, dataset)
    _split_validation(classes, N_validation_split, dataset)
    print('Downloaded dataset, converted .obj files and split off {} samples of each class for validation'.format(N_validation_split))


def _obj2vox(obj_path, size, double):
    npy_path = obj_path.replace('.obj', '_size{}.npy'.format(size))
    command = ["obj2voxel", "--size", str(size), obj_path, npy_path]
    if double:
        command.append('--double')
    subprocess.run(command)
    print('voxelized {}'.format(obj_path))
    # return npy_path
    # np.load(npy_path).astype(np.int8).reshape((size, size, size))

def obj2vox_conversion(size, double, root_dir, dataset):
    '''
    voxelize .obj samples
    :param size: grid size on which the dataset is voxelized
    :param double: sample on twice the grid size, then sample down to size
                   grid values are in 0,..,7 and represent number of face intersections in finer grid
    :param root_dir: dataset root directory
    :param dataset: dataset to be used (str for the dataset directory, either ModelNet10 or ModelNet40)
    '''

    obj_paths = sorted(glob.glob(os.path.join(root_dir, dataset, "*", "*", "*.obj")))
    assert len(obj_paths) != 0, 'no files found for conversion'
    Parallel(n_jobs=-1)(delayed(_obj2vox)(obj_path, size, double) for obj_path in obj_paths)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", default=False,
                        help="Download the dataset and convert to .obj, overwrite if it already exists (default: %(default)s)")
    parser.add_argument("--voxelize", action="store_true", default=False,
                        help="Voxelize the dataset (default: %(default)s)")
    parser.add_argument("--dataset", choices=['ModelNet10', 'ModelNet40'], default="ModelNet10",
                        help="Dataset to download and convert (default: %(default)s)")
    parser.add_argument('--dataset_root_dir', default='./data/', type=str,
                        help="root directory in which to save the dataset")
    parser.add_argument('--N_validation_split', default=20, type=int,
                        help="Number of samples to split from each class of the training set to the validation set")
    parser.add_argument('--size', type=int, required=True,
                        help="Side length of the voxel cube")
    parser.add_argument('--no_double_sampling', action='store_true', default=False,
                        help="Switch to NOT sampling on double resolution grid before downsampling (default: %(default)s)")

    args, unparsed = parser.parse_known_args()
    assert len(unparsed) == 0, 'unparsed arguments'

    if args.download:
        print('Downloading and converting dataset to .obj')
        if args.dataset == 'ModelNet10':
            url_data = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
            classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        elif args.dataset == 'ModelNet40':
            raise NotImplementedError('url and classes for ModelNet40 need to be added')
            # url_data = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet40.zip'
            # classes = 
        else:
            raise ValueError('unknown dataset argument')
        generate_obj_dataset(url_data, args.dataset_root_dir, classes, args.N_validation_split, args.dataset)

    if args.voxelize:
        print('Voxelize the dataset')
        obj2vox_conversion(args.size, not args.no_double_sampling, args.dataset_root_dir, args.dataset)
        print('Done')
