import torch
from e3nn.util import CodeGenMixin
from e3nn.util.jit import compile_mode
from e3nn import o3


@compile_mode('script')
class Extract(CodeGenMixin, torch.nn.Module):
    def __init__(self, irreps_in, irreps_outs, instructions):
        r"""Extract sub sets of irreps

        Parameters
        ----------
        irreps_in : `Irreps`
            representation of the input

        irreps_outs : list of `Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int


        Examples
        --------

        >>> c = Extract('1e + 0e + 0e', ['0e', '0e'], [(1,), (2,)])
        >>> c(torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
        (tensor([1.]), tensor([2.]))
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions

        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)

        code_out = [
            "import torch",
            "@torch.jit.script",
            "def main(x: torch.Tensor):",
            "    out = ("
        ]
        s = " "*4
        for irreps in self.irreps_outs:
            code_out.append(
                f"{s}{s}x.new_zeros(x.shape[:-1] + ({irreps.dim},)),"
            )
        code_out.append(f"{s})")  # close the out

        for i, (irreps_out, ins) in enumerate(zip(self.irreps_outs, self.instructions)):
            if ins == tuple(range(len(self.irreps_in))):
                code_out.append(
                    f"{s}out[{i}][:] = x"
                )
            else:
                for (s_out, _), i_in in zip(irreps_out.slices(), ins):
                    i_in1 = self.irreps_in[:i_in].dim
                    i_in2 = self.irreps_in[:i_in + 1].dim
                    code_out.append(
                        f"{s}out[{i}][..., {s_out.start}:{s_out.stop}] = x[..., {i_in1}:{i_in2}]"
                    )

        code_out.append(f"{s}return out")
        code_out = "\n".join(code_out)
        self._codegen_register({'_compiled_main_out': code_out})

    def forward(self, x):
        return self._compiled_main_out(x)


@compile_mode('script')
class ExtractIr(torch.nn.Module):
    def __init__(self, irreps_in, ir):
        r"""Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `Irreps`
            representation of the input

        ir : list of `Irrep`
            representation to extract
        """
        super().__init__()
        ir = o3.Irrep(ir)
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps([mul_ir for mul_ir in self.irreps_in if mul_ir.ir == ir])
        instructions = [tuple(i for i, mul_ir in enumerate(self.irreps_in) if mul_ir.ir == ir)]

        self.ext = Extract(self.irreps_in, [self.irreps_out], instructions)

    def forward(self, x):
        return self.ext(x)[0]
