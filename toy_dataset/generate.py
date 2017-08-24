#pylint: disable=E1101,C
import csv
import os
import numpy as np

i_csv = "../../datasets/shapenet/val.csv"
i_zip = "../../datasets/shapenet/val_normal.zip"
i_dir =                         "val_normal"

o_dir = "../../datasets/shapenet/voxels/val_normal"

try:
    os.mkdir(o_dir)
except:
    pass

with open(i_csv, "rt") as file:
    reader = csv.reader(file)
    rows = [row for row in reader]


id_ = rows[0].index("id")
cl_ = rows[0].index("synsetId")



for i, row in enumerate(rows[1:]):
    print("{}/{}    ".format(i, len(rows) - 1), end="\r")

    output = os.path.join(o_dir, row[id_] + ".npz")

    if not os.path.isfile(output):
        os.system("unzip -p {} {}/{}.obj > tmp.obj".format(i_zip, i_dir, row[id_]))

        os.system("obj2voxel --size 64 tmp.obj tmp.npy")

        x = np.load("tmp.npy")
        x = x.astype(np.int8)
        x = x.reshape((64, 64, 64))

        np.savez_compressed(output, x)
