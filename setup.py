# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import os
import re
import tarfile
import requests

from setuptools import find_packages, setup
import setuptools.command.install

from appdirs import user_cache_dir

import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# python setup.py develop    - if you wont to be able to execute from PyCharm (or similar IDE) - places .so file into e3nn folder from which real_spherical_harmonics imports

# Or:
# python setup.py build_ext
# python setup.py install    - PyCharm won't work, because it can't resolve import, but executable from terminal

if not torch.cuda.is_available():
    ext_modules = None
    print("GPU is not available. Skip building CUDA extensions.")
elif torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension('e3nn.real_spherical_harmonics',
                      sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                               'src/real_spherical_harmonics/rsh_cuda.cu'],
                      extra_compile_args={'cxx': ['-std=c++14'],
                                          'nvcc': ['-std=c++14']})
    ]
else:
    # GPU is available, but CUDA_HOME is None
    raise AssertionError("CUDA_HOME is undefined. Make sure nvcc compiler is available (cuda toolkit installed?)")


class PostInstallCommand(setuptools.command.install.install):
    """Post-installation for installation mode."""

    def run(self):
        setuptools.command.install.install.run(self)
        setuptools.command.install.install.do_egg_install(self)

        try:
            url = 'https://github.com/e3nn/e3nn/releases/download/v0.2-alpha/cache.tar'
            root = user_cache_dir("e3nn")

            if not os.path.isdir(root):
                os.makedirs(root)

            tar_path = os.path.join(root, "cache.tar")
            r = requests.get(url)
            open(tar_path, 'wb').write(r.content)

            tar = tarfile.open(tar_path)
            tar.extractall(root)
            tar.close()
        except:
            pass


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


setup(
    name='e3nn',
    version=find_version("e3nn", "__init__.py"),
    description='Equivariant convolutional neural networks '
                'for the group E(3) of 3 dimensional rotations, translations, and mirrors.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/e3nn/e3nn',
    packages=find_packages(exclude=["tests.*", "tests"]),
    ext_modules=ext_modules,
    install_requires=[
        'lie_learn',
        'scipy',
        'torch',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.7',
    cmdclass={'build_ext': BuildExtension, 'install': PostInstallCommand},
    license="MIT",
    license_files="LICENSE.txt",
)
