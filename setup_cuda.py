# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension('e3nn.rsh',
                  sources=['src/rsh/rsh_bind.cpp',
                           'src/rsh/rsh_cuda.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']})
]

setup(
    name='e3nn',
    version="0.0.0",
    url='https://e3nn.org',
    install_requires=[
        'scipy',
        'lie_learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.7",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
)
