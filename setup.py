# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# TODO: resolve issue with putting rsh_cuda in submodule se3cnn.rsh_cuda
setup(
    name='se3cnn',
    url='https://github.com/mariogeiger/se3cnn',
    install_requires=[
        'scipy',
        'lie_learn',
    ],
    dependency_links=['https://github.com/AMLab-Amsterdam/lie_learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension(
            name='rsh_cuda',
            sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                     'src/real_spherical_harmonics/rsh_cuda.cu'],
            extra_compile_args={'cxx': ['-std=c++14'],
                                'nvcc': ['-std=c++14']})
    ],
    cmdclass={
      'build_ext': BuildExtension
    },
    packages=find_packages(),
)
