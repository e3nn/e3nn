# Install

## pytorch-geometric

First you have to install pytorch-geometric, here are the commands for torch 1.7.0 and no cuda support:

```
TORCH=1.7.0
CUDA=cpu
pip install torch==$TORCH torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-geometric
```

See [here](https://github.com/rusty1s/pytorch_geometric#installation) to get cuda support or newer versions, they are all compatible with e3nn-core.

## e3nn-core

For now the library is pure python, therefore the installation precedure is very simple:

```
python setup.py install
```

or

```
python setup.py develop
```

We plan to incude (optional to build) home-made cuda kernels to accelerate the bottleneck operations.
