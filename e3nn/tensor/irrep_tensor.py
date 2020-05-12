# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin
from e3nn import rs


class IrrepTensor():
    def __init__(self, tensor, Rs):
        Rs = rs.convention(Rs)
        if tensor.shape[-1] != rs.dim(Rs):
            raise ValueError("Last tensor dimension and Rs do not have same dimension.")
        self.tensor = tensor
        self.Rs = Rs
