# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# python setup.py develop    - if you wont to be able to execute from PyCharm (or similar IDE) - places .so file into se3cnn folder from which real_spherical_harmonics imports

# Or:
# python setup.py build_ext
# python setup.py install    - PyCharm won't work, because it can't resolve import, but executable from terminal

ext_modules = [
    CUDAExtension('se3cnn.real_spherical_harmonics',
                  sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                           'src/real_spherical_harmonics/rsh_cuda.cu'],
                  extra_compile_args={'cxx': ['-std=c++14'],
                                      'nvcc': ['-std=c++14']})
]

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
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
)
