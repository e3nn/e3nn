#pylint: disable=E1101,C
import csv
import os
import numpy as np

i_csv = "../../datasets/shapenet/train.csv"
i_zip = "../../datasets/shapenet/train_normal.zip"
i_dir =                         "train_normal"

o_dir = "../../datasets/shapenet/voxels/train_augmented"

try:
    os.mkdir(o_dir)
except FileExistsError:
    print("already exists")

with open(i_csv, "rt") as file:
    reader = csv.reader(file)
    rows = [row for row in reader]


id_ = rows[0].index("id")
cl_ = rows[0].index("synsetId")

for i, row in enumerate(rows[1:]):
    print("{}/{}    ".format(i, len(rows) - 1), end="\r")

    try:
        os.mkdir(os.path.join(o_dir, row[id_]))
    except FileExistsError:
        pass

    for j in range(8):
        file = os.path.join(o_dir, row[id_], str(j) + ".npz")

        if not os.path.isfile(file):
            assert os.system("unzip -p {} {}/{}.obj > tmp.obj".format(i_zip, i_dir, row[id_])) == 0

            assert os.system("obj2voxel --size 64 -r tmp.obj tmp.npy") == 0

            x = np.load("tmp.npy")
            x = x.astype(np.int8)
            x = x.reshape((64, 64, 64))

            np.savez_compressed(file, x)
