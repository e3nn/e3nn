import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode("unsupported")
class BatchNorm(nn.Module):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    instance : bool
        apply instance norm instead of batch norm
    """
    __constants__ = ['instance', 'normalization', 'irs', 'affine']
    
    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, reduce="mean", instance=False, normalization="component"):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        self.features = []
        
        if self.instance:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
        else:
            self.register_buffer("running_mean", torch.zeros(num_scalar))
            self.register_buffer("running_var", torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce
        irs = []
        for mul, ir in self.irreps:
            irs.append((mul, ir.dim, ir.is_scalar()))
        self.irs = irs

        assert normalization in ["norm", "component"], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input):
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        orig_shape = input.shape
        batch = input.shape[0]
        dim = input.shape[-1]
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if self.training and not self.instance:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for (mul, d, is_scalar) in self.irs:
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if is_scalar:
                if self.training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(self._roll_avg(self.running_mean[irm : irm + mul], field_mean))
                else:
                    field_mean = self.running_mean[irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance:
                if self.normalization == "norm":
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == "component":
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(self._roll_avg(self.running_var[irv : irv + mul], field_norm))
            else:
                field_norm = self.running_var[irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and is_scalar:
                bias = self.bias[ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        torch._assert(ix == dim, f"`ix` should have reached input.size(-1) ({dim}), but it ended at {ix}")

        if self.training and not self.instance:
            torch._assert(irm == self.running_mean.numel(), "irm == self.running_mean.numel()")
            torch._assert(irv == self.running_var.size(0), "irv == self.running_var.size(0)")
        if self.affine:
            torch._assert(iw == self.weight.size(0), "iw == self.weight.size(0)")
            torch._assert(ib == self.bias.numel(), "ib == self.bias.numel()")

        if self.training and not self.instance:
            if len(new_means) > 0:
                torch.cat(new_means, out=self.running_mean)
            if len(new_vars) > 0:
                torch.cat(new_vars, out=self.running_var)

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(orig_shape)
