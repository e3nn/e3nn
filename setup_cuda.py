# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pack import KWARGS

ext_modules = [
    CUDAExtension('e3nn.rsh',
                  sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                           'src/real_spherical_harmonics/rsh_cuda.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']})
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    **KWARGS
)
