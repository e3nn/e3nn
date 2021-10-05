# Euclidean neural networks
[![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn?branch=main)
[![DOI](https://zenodo.org/badge/237431920.svg)](https://zenodo.org/badge/latestdoi/237431920)

**[Documentation](https://docs.e3nn.org)** | **[Code](https://github.com/e3nn/e3nn)** | **[ChangeLog](https://github.com/e3nn/e3nn/blob/main/ChangeLog.md)** | **[Colab](https://colab.research.google.com/drive/1Gps7mMOmzLe3Rt_b012xsz4UyuexTKAf?usp=sharing)**

The aim of this library is to help the developement of [E(3)](https://en.wikipedia.org/wiki/Euclidean_group) equivariant neural networks.
It contains fundamental mathematical operations such as [tensor products](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html) and [spherical harmonics](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html).

![](https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif)

## Installation

**Important:** install pytorch and only then run the command

```
pip install --upgrade pip
pip install --upgrade e3nn
```

For details and optional dependencies, see [INSTALL.md](https://github.com/e3nn/e3nn/blob/main/INSTALL.md)

### Breaking changes
e3nn is under development.
It is recommanded to install using pip. The main branch is considered as unstable.
The second version number is incremented every time a breaking change is made to the code.
```
0.(increment when backwards incompatible release).(increment for backwards compatible release)
```

## Help
We are happy to help! The best way to get help on `e3nn` is to submit a [Question](https://github.com/e3nn/e3nn/issues/new?assignees=&labels=question&template=question.md&title=%E2%9D%93+%5BQUESTION%5D) or [Bug Report](https://github.com/e3nn/e3nn/issues/new?assignees=&labels=bug&template=bug-report.md&title=%F0%9F%90%9B+%5BBUG%5D).

## Want to get involved? Great!
If you want to get involved in and contribute to the development, improvement, and application of `e3nn`, introduce yourself in the [discussions](https://github.com/e3nn/e3nn/discussions/new).

## Code of conduct
Our community abides by the [Contributor Covenant Code of Conduct](https://github.com/e3nn/e3nn/blob/main/code_of_conduct.md).

## Citing
```
@software{e3nn,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Alby M. and
                  Benjamin Kurt Miller and
                  Wouter Boomsma and
                  Bradley Dice and
                  Kostiantyn Lapchevskyi and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Simon Batzner and
                  Martin Uhrin and
                  Jes Frellsen and
                  Nuri Jung and
                  Sophia Sanborn and
                  Josh Rackers and
                  Michael Bailey},
  title        = {Euclidean neural networks: e3nn},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.4.0},
  doi          = {10.5281/zenodo.5292912},
  url          = {https://doi.org/10.5281/zenodo.5292912}
}
```

### Copyright

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy),
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin
and Kostiantyn Lapchevskyi. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
