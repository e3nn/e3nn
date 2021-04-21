# Install

## Dependencies

### PyTorch

e3nn requires PyTorch >=1.8.1 For installation instructions, please see the [PyTorch homepage](https://pytorch.org/).

### optional: torch_geometric

First you have to install [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric). For `torch` 1.8.1 and no CUDA support:

```bash
TORCH=1.8.0
CUDA=cpu

pip install --upgrade --force-reinstall torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --upgrade --force-reinstall torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --upgrade --force-reinstall torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install --upgrade --force-reinstall torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH+$CUDA.html
pip install torch-geometric
```

See [here](https://github.com/rusty1s/pytorch_geometric#installation) to get cuda support or newer versions.

### Optional: opt_einsum_fx (beta)

e3nn can use [`opt_einsum_fx`](https://github.com/Linux-cpp-lisp/opt_einsum_fx) to optimize the performance of `TensorProduct`s. To enable this, install `opt_einsum_fx`:
```bash
$ git clone https://github.com/Linux-cpp-lisp/opt_einsum_fx.git
$ cd opt_einsum_fx/
$ pip install .
```

`opt_einsum_fx` can be enabled/disabled using `e3nn.set_optimization_defaults(optimize_einsums=True/False)`. If you encounter any issues when `opt_einsum_fx` is enabled, please file an issue on the appropriate repository.

## e3nn

### Stable (PyPI)

```bash
$ pip install e3nn
```

### Unstable (Git)

```bash
$ git clone https://github.com/e3nn/e3nn.git
$ cd e3nn/
$ pip install .
```
