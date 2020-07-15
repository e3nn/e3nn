# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension
from setup_helper import BuildRDCExtension
from pack import KWARGS

ext_modules = [
    CUDAExtension('e3nn.real_spherical_harmonics',
                  sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                           'src/real_spherical_harmonics/rsh_cuda.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']}),
    CUDAExtension('e3nn.tensor_message',
                    sources=[
                            'src/tensor_message/link_tensor_message_cuda.cu',
                            'src/tensor_message/tensor_message_cuda.cu',
                            'src/tensor_message/tensor_message_bind.cpp'],
                    extra_compile_args={'nvcc': ['-std=c++14', '-rdc=true', '-lcudadevrt'],
                                        'cxx': ['-std=c++14']},
                    libraries=['cudadevrt'])
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildRDCExtension},
    **KWARGS
)
