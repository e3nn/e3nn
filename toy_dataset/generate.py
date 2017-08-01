#pylint: disable=E1101,C
import csv
import os
import numpy as np

i_csv = "../../datasets/shapenet/val.csv"
i_zip = "../../datasets/shapenet/val_perturbed.zip"
i_dir =                         "val_perturbed"

o_csv = "../../datasets/shapenet/voxels/val.csv"
o_dir = "../../datasets/shapenet/voxels/val_perturbed"

os.mkdir(o_dir)

with open(i_csv, "rt") as file:
    reader = csv.reader(file)
    rows = [row for row in reader]


id_ = rows[0].index("id")
cl_ = rows[0].index("synsetId")

classes = ["04379243", "03001627", "02691156", "02958343", "04256520"]

"""
03001627 = chair
02958343 = vehicules (cars, autocars)
02691156 = flying objects (airplanes, spaceships?)
04379243 = tables, desks, dresser, chair, ...
"""


subset = rows[:1] + [row for row in rows[1:] if row[cl_] in [cl for cl in classes]]

# save subset into csv file
with open(o_csv, "wt") as file:
    writer = csv.writer(file)
    writer.writerows(subset)

for i, row in enumerate(subset[1:]):
    print("{}/{}    ".format(i, len(subset) - 1), end="\r")

    os.system("unzip -p {} {}/{}.obj > tmp.obj".format(i_zip, i_dir, row[id_]))

    os.system("obj2voxel --size 64 --border 3 tmp.obj tmp.npy")

    x = np.load("tmp.npy")
    x = x.astype(np.int8)
    x = x.reshape((64, 64, 64))

    np.save(os.path.join(o_dir, row[id_] + ".npy"), x)
