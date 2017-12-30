# SE3 CNN

Functions to make SE(3) (3d rotations and translation) equivariant convolutional neural networks.
This is a *research* repository, not all the classes and function are relevant.

## Main elements

List of relevant classes defined in this repository.

- the class `SE3Convolution` defined in the file `se3_cnn/convolution.py`
- the class `BiasRelu` defined in the file `se3_cnn/non_linearities/scalar_activation.py`
- the class `SE3BatchNorm` defined in the file `se3_cnn/batchnorm.py`
- the classes `HighwayBlock` and `TensorProductBlock` defined in `se3_cnn/blocks/`

## Example

See file `arch/example.py`
