# Install

## pytorch-geometric

First you have to install pytorch-geometric, here are the commands for torch 1.8.0 and no cuda support:

```
TORCH=1.8.0
CUDA=cpu
pip install torch==$TORCH+$CUDA -f https://download.pytorch.org/whl/torch_stable.html

pip install scipy

pip install --no-index --upgrade --force-reinstall torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index --upgrade --force-reinstall torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index --upgrade --force-reinstall torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --no-index --upgrade --force-reinstall torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
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
