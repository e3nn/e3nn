from typing import Tuple

import torch
from torch import fx

from e3nn.util.jit import compile_mode
from e3nn import o3


@compile_mode('script')
class Extract(fx.GraphModule):
    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False):
        r"""Extract sub sets of irreps

        Parameters
        ----------
        irreps_in : `Irreps`
            representation of the input

        irreps_outs : list of `Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int

        squeeze_out : bool, default False
            if ``squeeze_out`` and only one output exists, a ``torch.Tensor`` will be returned instead of a ``Tuple[torch.Tensor]``


        Examples
        --------

        >>> c = Extract('1e + 0e + 0e', ['0e', '0e'], [(1,), (2,)])
        >>> c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
        (tensor([1.]), tensor([2.]))
        """
        super().__init__(self, fx.Graph())
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions

        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)

        # == generate code ==
        graph = self.graph
        x = fx.Proxy(graph.placeholder('x', torch.Tensor))
        torch._assert(x.shape[-1] == self.irreps_in.dim, "invalid input shape")

        out = []
        for irreps in self.irreps_outs:
            out.append(
                x.new_zeros(x.shape[:-1] + (irreps.dim,))
            )

        for i, (irreps_out, ins) in enumerate(zip(self.irreps_outs, self.instructions)):
            if ins == tuple(range(len(self.irreps_in))):
                out[i].copy_(x)
            else:
                for s_out, i_in in zip(irreps_out.slices(), ins):
                    i_start = self.irreps_in[:i_in].dim
                    i_len = self.irreps_in[i_in].dim
                    out[i].narrow(
                        -1, s_out.start, s_out.stop - s_out.start
                    ).copy_(
                        x.narrow(-1, i_start, i_len)
                    )

        out = tuple(e.node for e in out)
        if squeeze_out and len(out) == 1:
            graph.output(out[0], torch.Tensor)
        else:
            graph.output(out, Tuple[(torch.Tensor,)*len(self.irreps_outs)])

        self.recompile()


@compile_mode('script')
class ExtractIr(Extract):
    def __init__(self, irreps_in, ir):
        r"""Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `Irreps`
            representation of the input

        ir : `Irrep`
            representation to extract
        """
        ir = o3.Irrep(ir)
        irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)]

        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)
