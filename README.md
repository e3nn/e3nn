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

## Hierarchy

- `se3_cnn` contains the library
- `arch` contains architectures made with the library
- `notebook` contains [jupyer](http://jupyter.org/) notebooks
- `toy_dataset` contains scripts to generate a dataset by converting 3D models files (see [ModelNet](http://modelnet.cs.princeton.edu/) or [ShapeNet](https://www.shapenet.org/)) into voxels ugins [obj2voxel](https://github.com/antigol/obj2voxel)


## Dependencies

- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn)

## Usage

Install with
```
python setup.py install
```
