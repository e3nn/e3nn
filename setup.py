# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import os
import tarfile

import requests
import setuptools.command.install
from setuptools import find_packages, setup
# import torch
# from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from appdirs import user_cache_dir

# python setup.py develop    - if you wont to be able to execute from PyCharm (or similar IDE) - places .so file into e3nn folder from which real_spherical_harmonics imports

# Or:
# python setup.py build_ext
# python setup.py install    - PyCharm won't work, because it can't resolve import, but executable from terminal

# if not torch.cuda.is_available():
#     ext_modules = None
#     print("GPU is not available. Skip building CUDA extensions.")
# elif torch.cuda.is_available() and CUDA_HOME is not None:
#     ext_modules = [
#         CUDAExtension('e3nn.real_spherical_harmonics',
#                       sources=['src/real_spherical_harmonics/rsh_bind.cpp',
#                                'src/real_spherical_harmonics/rsh_cuda.cu'],
#                       extra_compile_args={'cxx': ['-std=c++14'],
#                                           'nvcc': ['-std=c++14']})
#     ]
# else:
#     # GPU is available, but CUDA_HOME is None
#     raise AssertionError("CUDA_HOME is undefined. Make sure nvcc compiler is available (cuda toolkit installed?)")


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


setup(
    name='e3nn',
    url='https://github.com/e3nn/e3nn',
    install_requires=[
        'scipy',
        'lie_learn',
        'appdirs'
    ],
    dependency_links=['https://github.com/AMLab-Amsterdam/lie_learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # ext_modules=ext_modules,
    cmdclass={
        # 'build_ext': BuildExtension,
        'install': PostInstallCommand
    },
    packages=find_packages(),
)
