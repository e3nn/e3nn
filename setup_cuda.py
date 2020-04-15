# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pack import KWARGS

ext_modules = [
    CUDAExtension('e3nn.cuda_rsh',
                  sources=['src/cuda_rsh/rsh_bind.cpp',
                           'src/cuda_rsh/rsh_cuda.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']})
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    **KWARGS
)
