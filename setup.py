#pylint: disable=C
import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

this_dir = os.path.dirname(__file__)

setup(
    name='se3_cnn',
    packages=find_packages(exclude=["build"]),
    ext_modules = cythonize("se3_cnn/*.pyx")
)
