# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `e3nn.nn.models.gate_points_2102` using node attributes along the length embedding to feed the radial network
- `Irreps.slices()`
- Module `Extract` (and `ExtractIr`) to extract subsets of irreps tensors
- Recursive TorchScript compiler `e3nn.util.jit`
- TorchScript support for `TensorProduct` and subclasses, `NormActivation`, `Gate`, `FullyConnectedNet`, and `gate_points_2101.Network`
### Removed
- `e3nn.math.reduce.reduce_tensor` in favor of `e3nn.o3.ReducedTensorProducts`
- swish, use `torch.nn.functional.silu` instead
- `"cartesian_vectors"` for equivariance testing — since the 0.2.2 Euler angle convention change, L=1 irreps are equivalent
### Fixed
- Modules that generate code now clean up their temporary files
- `NormActivation` now works on GPU

## [0.2.2] - 2021-02-09
### Changed
- Euler angle convention from ZYZ to YXY
- `TensorProduct.weight_shapes` content put into `TensorProduct.instructions`

### Added
- Better TorchScript support
