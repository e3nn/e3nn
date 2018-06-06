#pylint: disable=C
import matplotlib.pyplot as plt
import numpy as np


def main():
    steerable = np.array([
        [0, 1.0, 0.657388, 0.008196],
        [1, 2.0, 0.641843, 0.006038],
        [2, 4.0, 0.618149, 0.008430],
        [3, 8.0, 0.597872, 0.011057],
        [4, 16.0, 0.543758, 0.020094],
    ])

    cnn = np.array([
        [0, 1.0, 0.608836, 0.013043],
        [1, 2.0, 0.603203, 0.011121],
        [2, 4.0, 0.572544, 0.011519],
        [3, 8.0, 0.545515, 0.015933],
        [4, 16.0, 0.489087, 0.010964],
    ])

    plt.figure(figsize=(4, 2.8))

    plt.errorbar(steerable[:, 0], steerable[:, 2], yerr=steerable[:, 3], fmt=".", label="3D Steerable CNN", color="#190c9f")
    plt.errorbar(cnn[:, 0], cnn[:, 2], yerr=cnn[:, 3], fmt="x", label="3D CNN", color="#b02124")

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


main()
