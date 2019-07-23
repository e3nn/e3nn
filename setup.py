# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from setuptools import setup, find_packages

setup(
    name='se3cnn',
    url='https://github.com/mariogeiger/se3cnn',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'pytorch',
        'git+https://github.com/AMLab-Amsterdam/lie_learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
