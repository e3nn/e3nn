from typing import Optional, List, Union

import torch
import torch.fx

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.util import prod

from ._instruction import Instruction
from ._codegen import codegen_tensor_product


@compile_mode('script')
class TensorProduct(CodeGenMixin, torch.nn.Module):
    r"""Tensor product with parametrized paths.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        Irreps for the first input.

    irreps_in2 : `Irreps`
        Irreps for the second input.

    irreps_out : `Irreps`
        Irreps for the output.

    instructions : list of tuple
        List of instructions ``(i_1, i_2, i_out, mode, train[, path_weight])``.

        Each instruction puts ``in1[i_1]`` :math:`\otimes` ``in2[i_2]`` into ``out[i_out]``.

        * ``mode``: `str`. Determines the way the multiplicities are treated, ``"uvw"`` is fully connected. Other valid options are: ``'uvw'``, ``'uvu'``, ``'uvv'``, ``'uuw'``, ``'uuu'``, and ``'uvuv'``.
        * ``train``: `bool`. `True` if this path should have learnable weights, otherwise `False`.
        * ``path_weight``: `float`. A fixed multiplicative weight to apply to the output of this path. Defaults to 1. Note that setting ``path_weight`` breaks the normalization derived from ``in1_var``/``in2_var``/``out_var``.

    in1_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in1``. If ``None``, all default to ``1.0``.

    in2_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in2``. If ``None``, all default to ``1.0``.

    out_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_out``. If ``None``, all default to ``1.0``.

    normalization : {'component', 'norm'}
        The assumed normalization of the input and output representations. If it is set to "norm":

        .. math::

            \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1

    internal_weights : bool
        whether the `TensorProduct` contains its learnable weights as a parameter

    shared_weights : bool
        whether the learnable weights are shared among the input's extra dimensions

        * `True` :math:`z_i = w x_i \otimes y_i`
        * `False` :math:`z_i = w_i x_i \otimes y_i`

        where here :math:`i` denotes a *batch-like* index.
        ``shared_weights`` cannot be `False` if ``internal_weights`` is `True`.

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
    ...     "16x1o",
    ...     "16x1e",
    ...     [
    ...         (0, 0, 0, "uvw", True, 3),
    ...         (1, 0, 0, "uvw", True, 1),
    ...     ],
    ...     in2_var=[1/16]
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
    >>> with torch.no_grad():
    ...     for weight in module.weight_views():
    ...         mul_1, mul_2, mul_out = weight.shape
    ...         # formula from torch.nn.init.xavier_uniform_
    ...         a = (6 / (mul_1 * mul_2 + mul_out))**0.5
    ...         new_weight = torch.empty_like(weight)
    ...         new_weight.uniform_(-a, a)
    ...         weight[:] = new_weight
    tensor(...)
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
    _in1_dim: int
    _in2_dim: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        normalization: str = 'component',
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None
    ):
        # === Setup ===
        super().__init__()

        # Determine irreps
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if in1_var is None:
            self.in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            self.in1_var = [float(var) for var in in1_var]
            assert len(self.in1_var) == len(self.irreps_in1), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            self.in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            self.in2_var = [float(var) for var in in2_var]
            assert len(self.in2_var) == len(self.irreps_in2), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            self.out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            self.out_var = [float(var) for var in out_var]
            assert len(self.out_var) == len(self.irreps_out), "Len of out_var must be equal to len(irreps_out)"

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

        assert normalization in ['component', 'norm'], normalization
        self.normalization = normalization

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        opt_defaults = e3nn.get_optimization_defaults()
        self._specialized_code = _specialized_code if _specialized_code is not None else opt_defaults['specialized_code']
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults['optimize_einsums']
        del opt_defaults

        # Generate the actual tensor product code
        graph_out, graph_right, wigners = codegen_tensor_product(
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
            self._optimize_einsums
        )
        self._codegen_register({
            "_compiled_main_out": graph_out,
            "_compiled_main_right": graph_right
        })

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
                if any(
                    (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
                    for ins in self.instructions
                )
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
    def right(self, y, weight: Optional[torch.Tensor] = None):
        r"""Partially evaluate :math:`w x \otimes y`.

        It returns an operator in the form of a tensor that can act on an arbitrary :math:`x`.

        For example, if the tensor product above is expressed as

        .. math::

            w_{ijk} x_i y_j \rightarrow z_k

        then the right method returns a tensor :math:`b_{ik}` such that

        .. math::

            w_{ijk} y_j \rightarrow b_{ik}
            x_i b_{ik} \rightarrow z_k

        The result of this method can be applied with a tensor contraction:

        .. code-block:: python

            torch.einsum("...ik,...i->...k", right, input)

        Parameters
        ----------
        y : `torch.Tensor`
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
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"

        with torch.autograd.profiler.record_function(self._profiling_str):
            real_weight = self._get_weights(weight)
            return self._compiled_main_right(y, real_weight, self._wigner_buf)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        r"""Evaluate :math:`w x \otimes y`.

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim)``

        y : `torch.Tensor`
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
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"

        with torch.autograd.profiler.record_function(self._profiling_str):
            real_weight = self._get_weights(weight)
            return self._compiled_main_out(x, y, real_weight, self._wigner_buf)

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""View of weights corresponding to ``instruction``.

        Parameters
        ----------
        instruction : int
            The index of the instruction to get a view on the weights for. ``self.instructions[instruction].has_weight`` must be ``True``.

        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        Returns
        -------
        `torch.Tensor`
            A view on ``weight`` or this object's internal weights for the weights corresponding to the ``instruction`` th instruction.
        """
        if not self.instructions[instruction].has_weight:
            raise ValueError(f"Instruction {instruction} has no weights.")
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        return weight.narrow(-1, offset, prod(ins.path_shape)).view(batchshape + ins.path_shape)

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False
    ):
        r"""Iterator over weight views for each weighted instruction.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(batchshape + ins.path_shape)
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight
        return

    def visualize(
        self,
        weight: Optional[torch.Tensor] = None,
        plot_weight: bool = True,
        aspect_ratio=1,
        ax=None
    ):  # pragma: no cover
        r"""Visualize the connectivity of this `TensorProduct`

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        plot_weight : `bool`, default True
            Whether to color paths by the sum of their weights.

        ax : ``matplotlib.Axes``, default None
            The axes to plot on. If ``None``, a new figure will be created.

        Returns
        -------
        (fig, ax)
            The figure and axes on which the plot was drawn.
        """
        import numpy as np

        def _intersection(x, u, y, v):
            u2 = np.sum(u**2)
            v2 = np.sum(v**2)
            uv = np.sum(u * v)
            det = u2 * v2 - uv**2
            mu = np.sum((u * uv - v * u2) * (y - x)) / det
            return y + mu * v

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches

        if ax is None:
            ax = plt.gca()

        fig = ax.get_figure()

        # hexagon
        verts = [
            np.array([np.cos(a * 2 * np.pi / 6), np.sin(a * 2 * np.pi / 6)])
            for a in range(6)
        ]
        verts = np.asarray(verts)

        # scale it
        assert aspect_ratio in ['auto'] or isinstance(aspect_ratio, (float, int))

        if aspect_ratio == 'auto':
            factor = 0.2 / 2
            min_aspect = 1 / 2
            h_factor = max(len(self.irreps_in2), len(self.irreps_in1))
            w_factor = len(self.irreps_out)
            if h_factor / w_factor < min_aspect:
                h_factor = min_aspect * w_factor
            verts[:, 1] *= h_factor * factor
            verts[:, 0] *= w_factor * factor

        if isinstance(aspect_ratio, (float, int)):
            factor = 0.1 * max(len(self.irreps_in2), len(self.irreps_in1), len(self.irreps_out))
            verts[:, 1] *= factor
            verts[:, 0] *= aspect_ratio * factor

        codes = [
            Path.MOVETO,
            Path.LINETO,

            Path.MOVETO,
            Path.LINETO,

            Path.MOVETO,
            Path.LINETO,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, zorder=2)
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

        # get weights
        if weight is None and not self.internal_weights:
            plot_weight = False
        elif plot_weight:
            with torch.no_grad():
                path_weight = []
                for ins_i, ins in enumerate(self.instructions):
                    if ins.has_weight:
                        this_weight = self.weight_view_for_instruction(ins_i, weight=weight)
                        path_weight.append(this_weight.pow(2).mean())
                    else:
                        path_weight.append(0)
                path_weight = np.asarray(path_weight)
                path_weight /= np.abs(path_weight).max()
        cmap = matplotlib.cm.get_cmap('Blues')

        for ins_index, ins in enumerate(self.instructions):
            y = _intersection(s_in1[ins.i_in1], c_in1, s_in2[ins.i_in2], c_in2)

            verts = []
            codes = []
            verts += [s_out[ins.i_out], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in1[ins.i_in1], y]
            codes += [Path.MOVETO, Path.LINETO]
            verts += [s_in2[ins.i_in2], y]
            codes += [Path.MOVETO, Path.LINETO]

            if plot_weight:
                color = cmap(path_weight[ins_index]) if ins.has_weight else 'black'
            else:
                color = 'green' if ins.has_weight else 'black'

            ax.add_patch(patches.PathPatch(
                Path(verts, codes),
                facecolor='none',
                edgecolor=color,
                alpha=0.5,
                ls='-',
                lw=1.5 * ins.path_weight / min(i.path_weight for i in self.instructions),
            ))

        # add labels
        padding = 3
        fontsize = 10

        def format_ir(mul_ir):
            if mul_ir.mul == 1:
                return f"${mul_ir.ir}$"
            return f"${mul_ir.mul} \\times {mul_ir.ir}$"

        for i, mul_ir in enumerate(self.irreps_in1):
            ax.annotate(
                format_ir(mul_ir),
                s_in1[i],
                horizontalalignment='right',
                textcoords='offset points',
                xytext=(-padding, 0),
                fontsize=fontsize
            )

        for i, mul_ir in enumerate(self.irreps_in2):
            ax.annotate(
                format_ir(mul_ir),
                s_in2[i],
                horizontalalignment='left',
                textcoords='offset points',
                xytext=(padding, 0),
                fontsize=fontsize
            )

        for i, mul_ir in enumerate(self.irreps_out):
            ax.annotate(
                format_ir(mul_ir),
                s_out[i],
                horizontalalignment='center',
                verticalalignment='top',
                rotation=90,
                textcoords='offset points',
                xytext=(0, -padding),
                fontsize=fontsize
            )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis('equal')
        ax.axis('off')

        return fig, ax


class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product

    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicities.

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
        **kwargs
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out, instr, **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    r"""Elementwise connected tensor product.

    .. math::

        z_u = x_u \otimes y_u

    where :math:`u` runs over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    filter_ir_out : iterator of `Irrep`, optional
        filter to select only specific `Irrep` of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`

    Examples
    --------
    Elementwise scalar product

    >>> ElementwiseTensorProduct("5x1o + 5x1e", "10x1e", ["0e", "0o"])
    ElementwiseTensorProduct(5x1o+5x1e x 10x1e -> 5x0o+5x0e | 10 paths | 0 weights)

    """
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        filter_ir_out=None,
        **kwargs
    ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]

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

                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [
                    (i, i, i_out, 'uuu', False)
                ]

        super().__init__(irreps_in1, irreps_in2, out, instr, **kwargs)


class FullTensorProduct(TensorProduct):
    r"""Full tensor product between two irreps.

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` run over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    filter_ir_out : iterator of `Irrep`, optional
        filter to select only specific `Irrep` of the output

    normalization : {'component', 'norm'}
        see `TensorProduct`
    """
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        filter_ir_out=None,
        **kwargs
    ):

        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:

                    if filter_ir_out is not None and ir_out not in filter_ir_out:
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

        super().__init__(irreps_in1, irreps_in2, out, instr, **kwargs)
