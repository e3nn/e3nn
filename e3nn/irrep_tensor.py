from e3nn import rs


class IrrepTensor():
    def __init__(self, tensor, Rs):
        if tensor.shape[-1] != rs.dim(Rs):
            raise ValueError("Last tensor dimension and Rs do not have same dimension.")
        self.tensor = tensor
        self.Rs = Rs