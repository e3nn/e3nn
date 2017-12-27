# SE3 CNN

This is a *research* repository, not all the classes and function are relevant, see [this section](Main elements).

functions to make SE(3) (3d rotations and translation) equivariant convolutional neural networks.

## Dependencies

- [util_cnn](https://github.com/antigol/util_cnn) only for time profiling and file caching

## Main elements

List of relevant classes defined in this repository.

- the class `SE3Convolution` defined in the file `se3_cnn/convolution.py`
- the class `BiasRelu` defined in the file `se3_cnn/non_linearities/scalar_activation.py`
- the class `SE3BatchNorm` defined in the file `se3_cnn/batchnorm.py`
