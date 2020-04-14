# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from setuptools import find_packages, setup

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
    packages=find_packages(),
)
