# Install

## Dependencies

### PyTorch

e3nn requires PyTorch >=2.2.0. For installation instructions, please see the [PyTorch homepage](https://pytorch.org/).

### optional: torch_geometric

First you have to install [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric). For `torch` 2.2 and no CUDA support:

```bash
CUDA=cpu
TORCH=2.2.0

pip install --upgrade --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade --force-reinstall torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

See [here](https://github.com/rusty1s/pytorch_geometric#installation) to get cuda support or newer versions.

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
