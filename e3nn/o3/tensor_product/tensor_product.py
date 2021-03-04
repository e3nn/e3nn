from typing import Optional, List, Union

import torch
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from e3nn.util import prod

from ._instruction import Instruction
from ._codegen import codegen_tensor_product


@compile_mode('script')
class TensorProduct(CodeGenMixin, torch.nn.Module):
    r"""Tensor Product with parametrizable paths

    Parameters
    ----------
    in1 : `Irreps` or list of tuple
        List of first inputs ``(multiplicity, irrep[, variance])``.

    in2 : `Irreps` or list of tuple
        List of second inputs ``(multiplicity, irrep[, variance])``.

    out : `Irreps` or list of tuple
        List of outputs ``(multiplicity, irrep[, variance])``.

    instructions : list of tuple
        List of instructions ``(i_1, i_2, i_out, mode, train[, path_weight])``
        it means: Put ``in1[i_1]`` :math:`\otimes` ``in2[i_2]`` into ``out[i_out]``

        * mode: determines the way the multiplicities are treated, "uvw" is fully connected
        * train: `True` of `False` if this path is weighed by a parameter
        * path weight: how much this path should contribute to the output

    normalization : {'component', 'norm'}
        the way it is assumed the representation are normalized. If it is set to "norm":

        .. math::

            \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1

    internal_weights : bool
        does the instance of the class contains the parameters

    shared_weights : bool
        are the parameters shared among the inputs extra dimensions

        * `True` :math:`z_i = w x_i \otimes y_i`
        * `False` :math:`z_i = w_i x_i \otimes y_i`

        where here :math:`i` denotes a *batch-like* index

    Examples
    --------
    Create a module that computes elementwise the cross-product of 16 vectors with 16 vectors :math:`z_u = x_u \wedge y_u`

    >>> module = TensorProduct(
    ...     "16x1o", "16x1o", "16x1e",
    ...     [
    ...         (0, 0, 0, "uuu", False)
    ...     ]
    ... )

    Now mix all 16 vectors with all 16 vectors to makes 16 pseudo-vectors :math:`z_w = \sum_{u,v} w_{uvw} x_u \wedge y_v`

    >>> module = TensorProduct(
    ...     [(16, (1, -1))],
    ...     [(16, (1, -1))],
    ...     [(16, (1,  1))],
    ...     [
    ...         (0, 0, 0, "uvw", True)
    ...     ]
    ... )

    With custom input variance and custom path weights:

    >>> module = TensorProduct(
    ...     "8x0o + 8x1o",
    ...     [(16, "1o", 1/16)],
    ...     "16x1e",
    ...     [
    ...         (0, 0, 0, "uvw", True, 3),
    ...         (1, 0, 0, "uvw", True, 1),
    ...     ]
    ... )

    Example of a dot product:

    >>> irreps = o3.Irreps("3x0e + 4x0o + 1e + 2o + 3o")
    >>> module = TensorProduct(irreps, irreps, "0e", [
    ...     (i, i, 0, 'uuw', False)
    ...     for i, (mul, ir) in enumerate(irreps)
    ... ])

    Implement :math:`z_u = x_u \otimes (\sum_v w_{uv} y_v)`

    >>> module = TensorProduct(
    ...     "8x0o + 7x1o + 3x2e",
    ...     "10x0e + 10x1e + 10x2e",
    ...     "8x0o + 7x1o + 3x2e",
    ...     [
    ...         # paths for the l=0:
    ...         (0, 0, 0, "uvu", True),  # 0x0->0
    ...         # paths for the l=1:
    ...         (1, 0, 1, "uvu", True),  # 1x0->1
    ...         (1, 1, 1, "uvu", True),  # 1x1->1
    ...         (1, 2, 1, "uvu", True),  # 1x2->1
    ...         # paths for the l=2:
    ...         (2, 0, 2, "uvu", True),  # 2x0->2
    ...         (2, 1, 2, "uvu", True),  # 2x1->2
    ...         (2, 2, 2, "uvu", True),  # 2x2->2
    ...     ]
    ... )

    Tensor Product using the xavier uniform initialization:

    >>> irreps_1 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_2 = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> irreps_out = o3.Irreps("5x0e + 10x1o + 1x2e")
    >>> # create a Fully Connected Tensor Product
    >>> module = o3.TensorProduct(
    ...     irreps_1,
    ...     irreps_2,
    ...     irreps_out,
    ...     [
    ...         (i_1, i_2, i_out, "uvw", True, mul_1 * mul_2)
    ...         for i_1, (mul_1, ir_1) in enumerate(irreps_1)
    ...         for i_2, (mul_2, ir_2) in enumerate(irreps_2)
    ...         for i_out, (mul_out, ir_out) in enumerate(irreps_out)
    ...         if ir_out in ir_1 * ir_2
    ...     ]
    ... )
    >>> ws = []
    >>> for ins in module.instructions:
    ...     if ins.has_weight:
    ...         weight = torch.empty(ins.path_shape)
    ...         mul_1, mul_2, mul_out = weight.shape
    ...         # formula from torch.nn.init.xavier_uniform_
    ...         a = (6 / (mul_1 * mul_2 + mul_out))**0.5
    ...         ws += [weight.uniform_(-a, a).view(-1)]
    >>> with torch.no_grad():
    ...     module.weight[:] = torch.cat(ws)
    >>> n = 1_000
    >>> vars = module(irreps_1.randn(n, -1), irreps_2.randn(n, -1)).var(0)
    >>> assert vars.min() > 1 / 3
    >>> assert vars.max() < 3
    """
    _specialized_code: bool
    _optimize_einsums: bool
    _profiling_str: str
    normalization: str
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    in1_var: List[float]
    in2_var: List[float]
    out_var: List[float]

    def __init__(
        self,
        in1,
        in2,
        out,
        instructions,
        normalization='component',
        internal_weights=None,
        shared_weights=None,
        _specialized_code=True,
    ):
        # === Setup ===
        super().__init__()

        assert normalization in ['component', 'norm'], normalization
        self.normalization = normalization

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        # Determine irreps
        try:
            in1 = o3.Irreps(in1)
        except AssertionError:
            pass
        try:
            in2 = o3.Irreps(in2)
        except AssertionError:
            pass
        try:
            out = o3.Irreps(out)
        except AssertionError:
            pass

        in1 = [x if len(x) == 3 else x + (1.0,) for x in in1]
        in2 = [x if len(x) == 3 else x + (1.0,) for x in in2]
        out = [x if len(x) == 3 else x + (1.0,) for x in out]

        self.irreps_in1 = o3.Irreps([(mul, ir) for mul, ir, _var in in1])
        self.irreps_in2 = o3.Irreps([(mul, ir) for mul, ir, _var in in2])
        self.irreps_out = o3.Irreps([(mul, ir) for mul, ir, _var in out])

        self.in1_var = [var for _, _, var in in1]
        self.in2_var = [var for _, _, var in in2]
        self.out_var = [var for _, _, var in out]

        # Preprocess instructions into objects
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1, i_in2, i_out, connection_mode, has_weight, path_weight,
                {
                    'uvw': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    'uvu': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uvv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uuw': (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    'uuu': (self.irreps_in1[i_in1].mul,),
                    'uvuv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]
        self.instructions = instructions

        self._specialized_code = _specialized_code
        self.optimal_batch_size = None

        # Generate the actual tensor product code
        wigners = self._make_lazy_codegen()
        self._wigners = wigners

        # === Determine weights ===
        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)

        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor())

        # w3j
        wigner_mats = []
        for l_1, l_2, l_out in wigners:
            wig = o3.wigner_3j(l_1, l_2, l_out)

            if normalization == 'component':
                wig *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

            wigner_mats.append(wig)

        if len(wigner_mats) > 0:
            self.register_buffer('_wigner_buf', torch.cat([w.reshape(-1) for w in wigner_mats]))
        else:
            # We register an empty buffer so that call signatures don't have to change
            self.register_buffer('_wigner_buf', torch.Tensor())

        if self.irreps_out.dim > 0:
            output_mask = torch.cat([
                torch.ones(mul * ir.dim)
                if any(i.i_out == i_out and i.path_weight > 0 for i in self.instructions)
                else torch.zeros(mul * ir.dim)
                for i_out, (mul, ir) in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

        # For TorchScript, this needs to be done in advance:
        self._profiling_str = str(self)

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )

    @torch.jit.ignore
    def _make_lazy_codegen(self, compile: bool = True):
        lazygen_out, lazygen_right, wigners = codegen_tensor_product(
            self.irreps_in1,
            self.in1_var,
            self.irreps_in2,
            self.in2_var,
            self.irreps_out,
            self.out_var,
            self.instructions,
            self.normalization,
            self.shared_weights,
            self._specialized_code,
        )

        self._codegen_register(
            {
                '_compiled_main_out': lazygen_out,
                '_compiled_main_right': lazygen_right,
            },
            compile=compile
        )

        return wigners

    def __setstate__(self, d):
        # Set the dict with CodeGenMixin
        super().__setstate__(d)
        # Rebuild the lazy code generators
        # We don't compile with the new code generators — CodeGenMixin has already restored whatever exact code was compiled when the object was saved, and we want to preserve that.
        wigners = self._make_lazy_codegen(compile=False)
        assert wigners == self._wigners, "The provided saved state is inconsistant or from an incompatible version of e3nn"

    @torch.jit.unused
    def _prep_weights_python(self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [ins.path_shape for ins in self.instructions if ins.has_weight]
            if not self.shared_weights:
                weight = [w.reshape(-1, prod(shape)) for w, shape in zip(weight, weight_shapes)]
            else:
                weight = [w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            # If we're not scripting, then we're in Python and `weight` could be a List[Tensor]
            # deal with that:
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when the TensorProduct does not have `internal_weights`")
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert weight.ndim > 1, "When shared weights is false, weights must have batch dimension"
            return weight

    @torch.jit.export
    def right(self, features_2, weight: Optional[torch.Tensor] = None):
        r"""evaluate partially :math:`w x \cdot \otimes y`

        It returns an operator in the form of a matrix.

        Parameters
        ----------
        features_2 : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``weight_shape`` / ``(...) + weight_shape``.
            Use ``self.instructions`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim, irreps_out.dim)``
        """
        with torch.autograd.profiler.record_function(self._profiling_str):
            real_weight = self._get_weights(weight)
            return self._compiled_main_right(features_2, real_weight, self._wigner_buf)

    def forward(self, features_1, features_2, weight: Optional[torch.Tensor] = None):
        r"""evaluate :math:`w x \otimes y`

        Parameters
        ----------
        features_1 : `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim)``

        features_2 : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``

        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``weight_shape`` / ``(...) + weight_shape``.
            Use ``self.instructions`` to know what are the weights used for.

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        with torch.autograd.profiler.record_function(self._profiling_str):
            real_weight = self._get_weights(weight)
            return self._compiled_main_out(features_1, features_2, real_weight, self._wigner_buf)

    def visualize(self):  # pragma: no cover
        import numpy as np

        def _intersection(x, u, y, v):
            u2 = np.sum(u**2)
            v2 = np.sum(v**2)
            uv = np.sum(u * v)
            det = u2 * v2 - uv**2
            mu = np.sum((u * uv - v * u2) * (y - x)) / det
            return y + mu * v

        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches

        fig, ax = plt.subplots()

        # hexagon
        verts = [
            np.array([np.cos(a * 2 * np.pi / 6), np.sin(a * 2 * np.pi / 6)])
            for a in range(6)
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,

            Path.MOVETO,
            Path.LINETO,

            Path.MOVETO,
            Path.LINETO,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1)
        ax.add_patch(patch)

        n = len(self.irreps_in1)
        b, a = verts[2:4]

        c_in1 = (a + b) / 2
        s_in1 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        n = len(self.irreps_in2)
        b, a = verts[:2]

        c_in2 = (a + b) / 2
        s_in2 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        n = len(self.irreps_out)
        a, b = verts[4:6]

        s_out = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

        for ins in self.instructions:
            y = _intersection(s_in1[ins.i_in1], c_in1, s_in2[ins.i_in2], c_in2)

            verts = []
            codes = []
            verts += [s_out[ins.i_out], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in1[ins.i_in1], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in2[ins.i_in2], y]
            codes += [Path.MOVETO, Path.LINETO]

            ax.add_patch(patches.PathPatch(
                Path(verts, codes),
                facecolor='none',
                edgecolor='red' if ins.has_weight else 'black',
                alpha=0.5,
                ls='-',
                lw=ins.path_weight / min(i.path_weight for i in self.instructions),
            ))

        for i, ir in enumerate(self.irreps_in1):
            ax.annotate(ir, s_in1[i], horizontalalignment='right')

        for i, ir in enumerate(self.irreps_in2):
            ax.annotate(ir, s_in2[i], horizontalalignment='left')

        for i, ir in enumerate(self.irreps_out):
            ax.annotate(ir, s_out[i], horizontalalignment='center', verticalalignment='top', rotation=90)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis('equal')
        ax.axis('off')


class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicites.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    irreps_out : `Irreps`
        representation of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`

    internal_weights : bool
        see `TensorProduct`

    shared_weights : bool
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            irreps_out,
            normalization='component',
            internal_weights=None,
            shared_weights=None
                ):
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()

        instr = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out, instr, normalization, internal_weights, shared_weights)


class ElementwiseTensorProduct(TensorProduct):
    r"""Elementwise-Connected tensor product

    .. math::

        z_u = x_u \otimes y_u

    where :math:`u` runs over the irrep note that ther is no weights.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    set_ir_out : iterator of `Irrep`, optional
        representations of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            set_ir_out=None,
            normalization='component',
                ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if set_ir_out is not None:
            set_ir_out = [o3.Irrep(ir) for ir in set_ir_out]

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instr = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:

                if set_ir_out is not None and ir not in set_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [
                    (i, i, i_out, 'uuu', False)
                ]

        super().__init__(irreps_in1, irreps_in2, out, instr, normalization, internal_weights=False)


class FullTensorProduct(TensorProduct):
    r"""Full tensor product between two irreps

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` runs over the irrep, note that ther is no weights.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    set_ir_out : iterator of `Irrep`, optional
        representations of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`
    """
    def __init__(
            self,
            irreps_in1,
            irreps_in2,
            set_ir_out=None,
            normalization='component',
                ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if set_ir_out is not None:
            set_ir_out = [o3.Irrep(ir) for ir in set_ir_out]

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:

                    if set_ir_out is not None and ir_out not in set_ir_out:
                        continue

                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr += [
                        (i_1, i_2, i_out, 'uvuv', False)
                    ]

        out = o3.Irreps(out)
        out, p, _ = out.sort()

        instr = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instr
        ]

        super().__init__(irreps_in1, irreps_in2, out, instr, normalization, internal_weights=False)
