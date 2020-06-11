import os
import re
from setuptools import find_packages

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


KWARGS = dict(
    name='e3nn',
    version=find_version("e3nn", "__init__.py"),
    description='Equivariant convolutional neural networks '
                'for the group E(3) of 3 dimensional rotations, translations, and mirrors.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://e3nn.org',
    packages=find_packages(exclude=["tests.*", "tests"]),
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
    python_requires='>=3.7',
    license="MIT",
    license_files="LICENSE.txt",
)
