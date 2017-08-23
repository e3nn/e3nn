#pylint: disable=E1101,C
import csv
import random

in_csv = "../../datasets/shapenet/voxels/val.csv"
classes = ["02691156", "02958343", "04256520", "04090263", "03636649", "04530566"]

"""
04379243 = tables, desks, dresser, chair, ... (5876 objects)
03001627 = chair (modern style) (4612 objects)
02691156 = flying objects (airplanes, spaceships?) (2832 objects)
02958343 = vehicules (cars, autocars) (2502 objects)
04256520 = chair (simple style) (2198 objects)
04090263 = gun (1655 objcets)
03636649 = lamps (1620 objects)
04530566 = boat (1356 objects)
"""

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
