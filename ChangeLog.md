# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- `Extract` uses `CodeGenMixin` to avoid strange recursion errors during training
- Add missing call to `normalize` in `axis_angle_to_quaternion`

## [0.3.4] - 2021-08-20
### Fixed
- `ReducedTensorProducts`: `normalization` and `filter_ir_mid` where not properly propagated through the recusive calls, this bug has no effects if the default values where used
- Use `torch.linalg.eigh` instead of the deprecated `torch.symeig`

### Added
- (dev only) Pre-commit hooks that run pylint and flake8.  These catch some common mistakes/style issues.
- classes to do `SO(3)` Grid transform (not fast) and Activation function using it
- Add `f_in` and `f_out` to `o3.Linear`
- `PBC` guide in the doc

## [0.3.3] - 2021-06-21
### Changed
- `FullyConnectedNet` is now a `torch.nn.Sequential`

### Fixed
- `BatchNorm` was not equivariant for pseudo-scalars

### Added
- `biases` argument to `o3.Linear`
- `nn.models.v2106`: `MessagePassing` takes a sequence of irreps
- `nn.models.v2106`: `Convolution` inpired from [Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks](`https://arxiv.org/pdf/2002.10444.pdf`)

## [0.3.2] - 2021-06-10
### Added
- [`opt_einsum_fx`](https://github.com/Linux-cpp-lisp/opt_einsum_fx) as a dependency
- `p=-1` option for `Irreps.spherical_harmonics(lmax, p)`

### Removed
- Removed `group/_linalg` (`has_rep_in_rep` and `intertwiners`) (should use `equivariant-MLP` instead)

## [0.3.1] - 2021-05-26
### Added
- `preprocess` function in `e3nn.nn.models.v2103.gate_points_networks.SimpleNetwork`
- Specialized code for `mode="uuw"`
- `instance` argument to `nn.BatchNorm`

## [0.3.0] - 2021-05-10
### Added
- `pool_nodes` argument (default `True`) to networks in `e3nn.nn.models.v2103.gate_points_networks`
- Instruction support for `o3.Linear`
- `o3.Linear.weight_views` and `o3.Linear.weight_view_for_instruction`
- `nn.Dropout`

### Changed
- `o3.Linear` and `o3.FullyConnectedTensorProduct` no longer automatically simplifies its `irreps_in` or `irreps_out`. If you want this behaviour, simplify your irreps explicitly!

### Fixed
- `TensorProduct` can now gracefully handle multiplicities of zero
- `weight_views`/`weight_view_for_instruction` methods now support `shared_weights=False`

## [0.2.9] - 2021-05-04
### Added
- Normalization testing with `assert_normalized`
- Optional logging for equivariance and normalization tests
- Public `e3nn.util.test.format_equivariance_error` method for printing equivariance test results
- Module `o3.SphericalHarmonicsAlphaBeta`

### Changed
- Generated code (modules like `TensorProduct`, `Linear`, `Extract`) now pickled using TorchScript IR, rather than Python source code.
- e3nn now only requires PyTorch >= 1.8.0 rather than 1.8.1
- Changed `o3.legendre` into a module `o3.Legendre`

### Removed
- Removed `e3nn.util.codegen.eval_code` in favor of `torch.fx`

## [0.2.8] - 2021-04-21
### Added
- `squared` option to `o3.Norm`
- `e3nn.nn.models.v2104.voxel_convolution.Convolution` made to be resolution agnostic
- `TensorProduct.visualize` keyword argument `aspect_ratio`

### Changed
- `ReducedTensorProducts` is a (scriptable) `torch.nn.Module`
- e3nn now requires the latest stable PyTorch, >=1.8.1
- `TensorProduct.visualize`: color of paths based on `w.pow(2).mean()` instead of `w.sum().sign() * w.abs().sum()`

### Fixed
- No more NaN gradients of `o3.Norm`/`nn.NormActivation` at zero when using `epsilon`
- Modules with `@compile_mode('trace')` can now be compiled when their dtype and the current default dtype are different
- Fix errors in `ReducedTensorProducts` and add new tests

## [0.2.7] - 2021-04-14
### Added
- `uuu` connection mode in `o3.TensorProduct` now has specialized code

### Fixed
- Fixed an issue with `Activation` (used by `Gate`). It was only applying the first activation function provided. `Activation('0e+0e', [act1, act2])` was equivalent to `Activation('2x0e', [act1])`. Solved by removing the `.simplify()` applied to `self.irreps_in`.
- `Gate` will not accept non-scalar `irreps_gates` or `irreps_scalars`

## [0.2.6] - 2021-04-12
### Added
- `e3nn.util.test.random_irreps` convinience function for writing tests

### Changed
- `o3.Linear` now has more efficient specialized code

### Fixed
- Fixed a problem with temporary files on windows

## [0.2.5] - 2021-04-07
### Added
- Added `e3nn.set_optimization_defaults()` and `e3nn.get_optimization_defaults()`
- Constructors for empty `Irreps`: `Irreps()` and `Irreps("")`
- Additional tests, docs, and refactoring for `Irrep` and `Irreps`.
- Added `TensorProduct.weight_views()` and `TensorProduct.weight_view_for_instruction()`
- Fix Docs for ExtractIr

### Changed
- Renamed `o3.TensorProduct` arguments in `irreps_in1`, `irreps_in2` and `irreps_out`
- Renamed `o3.spherical_harmonics` arguement `xyz` into `x`
- Renamed `math.soft_one_hot_linspace` argument `endpoint` into `cutoff`, `cutoff = not endpoint`
- Variances are now provided to `o3.TensorProduct` through explicit `in1_var`, `in2_var`, `out_var` parameters
- Submodules define `__all__`; documentation uses shorter module names for the classes/methods.

### Fixed
 - Enabling/disabling einsum optimization no longer affects PyTorch RNG state.

### Removed
- Variances can no longer be provided to `o3.TensorProduct` in the list-of-tuple format for `irreps_in1`, etc.

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
- Optional optimization of `TensorProduct` if [`opt_einsum_fx`](https://github.com/Linux-cpp-lisp/opt_einsum_fx) is installed

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
