import os
import re
from setuptools import find_packages
from torch.utils.cpp_extension import CUDAExtension
from setup_helper import BuildRDCExtension

# Recommendations from https://packaging.python.org/
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


ext_modules = [
    CUDAExtension('e3nn.real_spherical_harmonics',
                  sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                           'src/real_spherical_harmonics/rsh.cu',
                           'src/real_spherical_harmonics/e3nn_normalization.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']}),
    CUDAExtension('e3nn.tensor_message',
                    sources=['src/tensor_message/link_tensor_message_cuda.cu',
                            'src/tensor_message/tensor_message_cuda.cu',
                            'src/tensor_message/tensor_message_bind.cpp'],
                    extra_compile_args={'nvcc': ['-std=c++14', '-rdc=true', '-lcudadevrt'],
                                        'cxx': ['-std=c++14']},
                    libraries=['cudadevrt']),
    CUDAExtension('e3nn.tensor_block_cuda',
                    sources=['src/tensor_block/link_tensor_block_cuda.cu',
                             'src/tensor_block/tensor_block_cuda.cu',
                             'src/tensor_block/tensor_block_bind.cpp'],
                    extra_compile_args={'nvcc': ['-std=c++14', '-rdc=true', '-lcudadevrt'],
                                        'cxx': ['-std=c++14']},
                    libraries=['cudadevrt'])
]

KWARGS = dict(
    name='e3nn',
    version=find_version("e3nn", "__init__.py"),
    description='Equivariant convolutional neural networks '
                'for the group E(3) of 3 dimensional rotations, translations, and mirrors.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://e3nn.org',
    packages=find_packages(exclude=["tests.*", "tests"]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildRDCExtension},
    install_requires=[
        'lie_learn',
        'scipy',
        'sympy',
        'torch>=1.4.0',
        'torch_scatter',
        'torch_sparse',
        'torch_cluster',
        'torch_spline_conv',
        'torch_geometric',
        'ase',
        'pymatgen',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.6',
    license="MIT",
    license_files="LICENSE.txt",
)
