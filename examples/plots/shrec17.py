import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np

plt.figure(figsize=(2.8, 2.5))

data = [
    ['Furuya', 1.132, 8.4e6],
    ['Esteves', 1.129, 0.5e6],
    ['Tatsuma', 1.114, 3e6],
    ['Ours', 1.110, 142e3],
    ['Zhou', 0.973, 36e6],
    ['Kanezaki', 0.933, 61100840],
    ['Deng', 0.853, 138357544],
]

plt.scatter([p for n, a, p in data], [a for n, a, p in data], c="#190c9f")
plt.xscale('log')
plt.xlabel('number of parameters')
plt.ylabel('micro mAP + macro mAP')
plt.xlim(1e5, 3e8)
plt.ylim(0.75, 1.15)

for n, a, p in data:
    plt.annotate(n, (p, a - 0.02), rotation=-90,
    horizontalalignment='center', verticalalignment='top')

plt.tight_layout()
plt.savefig("shrec17.pgf", transparent=True)

