#pylint: disable=E1101,C
import csv
import random

in_csv = "../../datasets/shapenet/voxels/val.csv"
classes = ["02691156", "02958343", "04256520", "04090263", "03636649", "04530566"]
how_much = 194
out_csv = "../../datasets/shapenet/voxels/val_{}x{}.csv".format(len(classes), how_much)

with open(in_csv, "rt") as file:
    reader = csv.reader(file)
    rows = [r for r in reader]


result = []
for c in classes:
    x = [r for r in rows if r[1] == c]
    random.shuffle(x)
    if len(x) < how_much:
        raise ValueError("not enough files, only {} for class {}".format(len(x), c))
    x = x[:how_much]

    result.extend(x)

with open(out_csv, "wt") as file:
    writer = csv.writer(file)
    writer.writerows(result)
