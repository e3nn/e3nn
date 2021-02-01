# Install

## pytorch-geometric

First you have to install pytorch-geometric, here are the commands for torch 1.7.1 and no cuda support:

```
TORCH=1.7.1
CUDA=cpu

pip install torch==$TORCH torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

TORCH=1.7.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-geometric
```

See [here](https://github.com/rusty1s/pytorch_geometric#installation) to get cuda support or newer versions.

## e3nn

For now the library is pure python, therefore the installation precedure is very simple:

```
pip install e3nn
```

or clone the main branch and execute

```
python setup.py develop
```
