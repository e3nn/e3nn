import os
import csv
import numpy as np

n = 64

for i in range(n):
    os.system("obj2voxel --size 64 --rotate hand.obj tmp.npy")
    x = np.load("tmp.npy").astype(np.int8).reshape((64, 64, 64))
    if i % 2 == 1:
        x = x[::-1]
    np.save("data/"+str(i)+".npy", x)
os.system("rm tmp.npy")

writer = csv.writer(open("train.csv", "tw"))
for i in range(n // 2):
    writer.writerow([i, i % 2])

writer = csv.writer(open("val.csv", "tw"))
for i in range(n // 2, n):
    writer.writerow([i, i % 2])
