# Install e3nn

If you don't need cuda kernels acceleration simply execute
```
python setup.py install
```

# How to compile the CUDA kernels

First you need to install `nvcc` and `conda`.

## Create a new python environment
```
conda create -n e3nn python=3.8
```

To activate the environement execute
```
conda activate e3nn
```

## Install pytorch

pytorch need to be compatible with you version of `nvcc`.
Check your version of nvcc
```
nvcc --version
```

If like me you have `Cuda compilation tools, release 9.2, V9.2.148` install pytorch with
```
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
**The version of cuda needs to match between pytorch and nvcc!**

Now we can check if cuda works
```
python -c "import torch; print(torch.randn((), device='cuda'))"
```

## Compile the code

Clone this repository
```
git clone git@github.com:e3nn/e3nn.git
cd e3nn
```

Remove previous attempts and cache
```
rm -rf build dist e3nn.egg-info __pycache__
rm -rf .cache/pip
```

Execute installation script
```
python setup_cuda.py install
```

Check compilation success
```
python -c "import torch; import e3nn.cuda_rsh"
```

