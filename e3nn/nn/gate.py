import torch

from e3nn import o3
from e3nn.nn import Extract
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode


@compile_mode('trace')
class Activation(torch.nn.Module):
    r"""Scalar activation function

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------

    >>> a = Activation("256x0o", [torch.abs])
    >>> a.irreps_out
    256x0e

    >>> a = Activation("256x0o+16x1e", [None, None])
    >>> a.irreps_out
    256x0o+16x1e
    """
    def __init__(self, irreps_in, acts):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        x = torch.linspace(0, 10, 256)

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError("Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.acts = acts

    def forward(self, features, dim=-1):
        '''evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        '''
        with torch.autograd.profiler.record_function(repr(self)):
            output = []
            index = 0
            for (mul, ir), act in zip(self.irreps_in, self.acts):
                if act is not None:
                    output.append(act(features.narrow(dim, index, mul)))
                else:
                    output.append(features.narrow(dim, index, mul * ir.dim))
                index += mul * ir.dim

            if output:
                return torch.cat(output, dim=dim)
            else:
                return torch.zeros_like(features)


@compile_mode('trace')
class _Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(o3.Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, o3.Irreps([]))

        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions += [tuple(range(i, i + len(irreps_out)))]
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)

        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions]

        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def forward(self, x):
        return self.cut(x)


@compile_mode('script')
class Gate(torch.nn.Module):
    r"""Gate activation function

    Parameters
    ----------
    irreps_scalars : `Irreps`
        representation of the scalars (the one not use for the gates)

    act_scalars : list of function or None
        activations acting on the scalars

    irreps_gates : `Irreps`
        representation of the scalars used to gate.
        ``irreps_gates.num_irreps == irreps_nonscalars.num_irreps``

    act_gates : list of function or None
        activations acting on the gates

    irreps_nonscalars : `Irreps`
        representation of the non-scalars.
        ``irreps_gates.num_irreps == irreps_nonscalars.num_irreps``

    Examples
    --------

    >>> g = Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
    >>> g.irreps_out
    16x0o+16x1o+16x1e
    """
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_nonscalars):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_nonscalars = o3.Irreps(irreps_nonscalars)

        self.sc = _Sortcut(irreps_scalars, irreps_gates)
        self.irreps_scalars, self.irreps_gates = self.sc.irreps_outs
        self.irreps_nonscalars = irreps_nonscalars.simplify()
        self.irreps_in = self.sc.irreps_in + self.irreps_nonscalars

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_nonscalars, irreps_gates)
        irreps_nonscalars = self.mul.irreps_out

        self.irreps_out = irreps_scalars + irreps_nonscalars

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        '''evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        '''
        with torch.autograd.profiler.record_function('Gate'):
            scalars, gates = self.sc(features)
            nonscalars = features[..., scalars.shape[-1] + gates.shape[-1]:]

            scalars = self.act_scalars(scalars)
            if gates.shape[-1]:
                gates = self.act_gates(gates)
                nonscalars = self.mul(nonscalars, gates)
                features = torch.cat([scalars, nonscalars], dim=-1)
            else:
                features = scalars
            return features
