# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the top!

## [Unreleased]
### Added
- Constructors for empty `Irreps`: `Irreps()` and `Irreps("")`

### Changed
- Renamed `o3.spherical_harmonics` arguement `xyz` into `x`
- Renamed `math.soft_one_hot_linspace` argument `endpoint` into `cutoff`, `cutoff = not endpoint`

## [0.2.4] - 2021-03-23
### Added
- `basis='smooth_finite'` option to `math.soft_one_hot_linspace`
- `math.soft_unit_step` function
- `nn.model.v2103` generic message passing model + examples of networks using it.
- `o3.TensorProduct`: is jit scriptable
- `o3.TensorProduct`: also broadcast the `weight` argument
- simple e3nn models can be saved/loaded with `torch.save()`/`torch.load()`
- JITable `o3.SphericalHarmonics` module version of `o3.spherical_harmonics`
- `in_place` option for `e3nn.util.jit` compilation functions
- New `@compile_mode("unsupported")` for modules that do not support TorchScript
- flake8 settings have been added to `setup.cfg` for improved code style
- `TensorProduct.visualize()` can now plot weights
- `basis='bessel'` option to `math.soft_one_hot_linspace`

### Changed
- `o3.TensorProduct` now uses `torch.fx` to generate it's code
- e3nn now requires the latest stable PyTorch, >=1.8.0
- in `soft_one_hot_linspace` the argument `base` is renamed into `basis`
- `Irreps.slices()`, do `zip(irreps.slices(), irreps)` to retrieve the old behavior
- `math.soft_one_hot_linspace` very small change in the normalization of `fourier` basis
- `normalize2mom` is now a `torch.nn.Module`
- rename arguments `set_ir_...` into `filter_ir_...`
- Renamed `e3nn.nn.Gate` argument `irreps_nonscalars` to `irreps_gated`
- Renamed `e3nn.o3.TensorProduct` arguments `x1, x2` to `x, y`

### Fixed
- `nn.Gate` was crashing when the number of scalars or gates was zero
- `device` edge cases for `Gate` and `SphericalHarmonics`

## [0.2.3] - 2021-02-23
### Added
- Add argument `basis` into `math.soft_one_hot_linspace` that can take values `gaussian`, `cosine` and `fourier`
- `io.SphericalTensor.sum_of_diracs`
- Optional arguments `function(..., device=None, dtype=None)` for many functions
- `e3nn.nn.models.gate_points_2102` using node attributes along the length embedding to feed the radial network
- `Irreps.slices()`
- Module `Extract` (and `ExtractIr`) to extract subsets of irreps tensors
- Recursive TorchScript compiler `e3nn.util.jit`
- TorchScript support for `TensorProduct` and subclasses, `NormActivation`, `Gate`, `FullyConnectedNet`, and `gate_points_2101.Network`

### Changed
- in `o3.TensorProduct.instructions`: renamed `weight_shape` in `path_shape` and is now set even if `has_weight` is `False`
- `o3.TensorProduct` weights are now flattened tensors
- rename `io.SphericalTensor.from_geometry_adjusted` into `io.SphericalTensor.with_peaks_at`
- in `ReducedTensorProducts`, `ElementwiseTensorProduct` and `FullTensorProduct`: rename `irreps_out` argument into `set_ir_out` to not confuse it with `o3.Irreps`

### Removed
- `io.SphericalTensor.from_geometry_global_rescale`
- `e3nn.math.reduce.reduce_tensor` in favor of `e3nn.o3.ReducedTensorProducts`
- swish, use `torch.nn.functional.silu` instead
- `"cartesian_vectors"` for equivariance testing — since the 0.2.2 Euler angle convention change, L=1 irreps are equivalent

### Fixed
- `io.SphericalTensor.from_samples_on_s2` manage batch dimension
- Modules that generate code now clean up their temporary files
- `NormActivation` now works on GPU

## [0.2.2] - 2021-02-09
### Changed
- Euler angle convention from ZYZ to YXY
- `TensorProduct.weight_shapes` content put into `TensorProduct.instructions`

### Added
- Better TorchScript support
