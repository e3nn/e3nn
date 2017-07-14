import csv
import os
import numpy as np

with open("../../datasets/shapenet/train.csv", "rt") as file:
    reader = csv.reader(file)
    rows = [row for row in reader]
    

id_ = rows[0].index("id")
cl_ = rows[0].index("synsetId")

classes = [("cars", "02958343"), ("airplane", "02691156")]

subset = rows[:1] + [row for row in rows[1:] if row[cl_] in [cl for _, cl in classes]]

# save subset into csv file
with open('voxels/train.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(subset)

for i, row in enumerate(subset[1:]):
    print("{}/{}    ".format(i, len(subset) - 1), end="\r")

    os.system("unzip -p ../../datasets/shapenet/train_normal.zip train_normal/{}.obj > tmp.obj".format(row[id_]))
    
    os.system("obj2voxel --size 64 --border 3 tmp.obj tmp.npy")
    
    x = np.load("tmp.npy")
    x = x.astype(np.int8)
    x = x.reshape((64, 64, 64))
    
    np.save(os.path.join("voxels/train", row[id_] + ".npy"), x)
