import matplotlib.pyplot as plt
import numpy as np


steerable = np.array([
    [4, 0.55],
    [2, 0.62],
    [3, 0.59],
    [4, 0.54],
    [0, 0.66],
    [2, 0.62],
    [3, 0.59],
    [1, 0.64],
    [1, 0.64],
    [0, 0.67],
])

cnn = np.array([
    [4, 0.48],
    [0, 0.61],
    [1, 0.60],
    [2, 0.58],
    [3, 0.53],
    [4, 0.48],
    [3, 0.55],
    [0, 0.62],
    [1, 0.61],
    [2, 0.56],
])

plt.figure(figsize=(4, 2.8))

plt.plot(steerable[:, 0], steerable[:, 1], ".", label="3D Steerable CNN", color="#190c9f")
plt.plot(cnn[:, 0], cnn[:, 1], "x", label="3D CNN", color="#b02124")


plt.xticks(range(5), [r"$2^{}$".format(i) for i in range(5)])
y = np.arange(0.5, 0.65, 0.05)
plt.yticks(y, [r"${:.2f}$".format(i) for i in y])

plt.xlabel("training set size reduction factor")
plt.ylabel("test accuracy")
plt.legend()
plt.grid(axis="y")

plt.tight_layout()

# plt.savefig("cath.pdf")
plt.savefig("cath.pgf")
