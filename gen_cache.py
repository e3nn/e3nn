"""
Generate the .cache files
"""
from e3nn import o3

lmax = 10

for l1 in range(lmax + 1):
    for l2 in range(lmax + 1):
        for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
            print(l1, l2, l3)
            o3.wigner_3j(l1, l2, l3)
