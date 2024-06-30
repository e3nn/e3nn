import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing
from e3nn import o3
from torch._inductor.utils import print_performance
from attr import attrs, attrib

from typing import Optional, List, Tuple


class IrrepsArray(torch.Tensor):

    @staticmethod
    def __new__(cls, array, irreps):
        kwargs = {}
        shape = array.shape
        kwargs["strides"] = array.stride()
        kwargs["storage_offset"] = array.storage_offset()
        kwargs["device"] = array.device
        kwargs["layout"] = array.layout
        kwargs["requires_grad"] = array.requires_grad
        kwargs["dtype"] = array.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
    
    def __init__(self,
                 array: torch.Tensor,
                 irreps: o3.Irreps,
                 _zero_flags: Optional[Tuple] = None):
        
        self.array = array
        self.irreps = irreps
        self._zero_flags = _zero_flags

    def __repr__(self):  # noqa: D105
        r = str(self.array)
        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(IrrepsArray, lambda x: x.array, args)

        kwargs_inner = pytree.tree_map_only(IrrepsArray, lambda x: x.aray, kwargs)

        out_inner = func(*args_inner, **kwargs_inner)
        out_inner_flat, spec = pytree.tree_flatten(out_inner)
        # for aten ops that return non-tensors, just assume that
        # our cust inner tensors return the same value
        out_flat = [
            IrrepsArray(o_inner) if isinstance(o_inner, torch.Tensor) else o_inner
            for o_inner in out_inner_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)
    
    def __post_init__(self):
        if hasattr(self.array, "shape"):
            if self.array.shape[-1] != self.irreps.dim:
                raise ValueError(
                    f"IrrepsArray: Array shape {self.array.shape} incompatible with irreps {self.irreps}. "
                    f"{self.array.shape[-1]} != {self.irreps.dim}"
                )
        if self.zero_flags is not None:
            if len(self.zero_flags) != len(self.irreps):
                raise ValueError(
                    f"IrrepsArray: len(zero_flags) != len(irreps), {len(self.zero_flags)} != {len(self.irreps)}"
                )

    @property
    def zero_flags(self):
        if self._zero_flags is None:
            return (False,) * len(self.irreps)
        return self._zero_flags

    @property
    def chunks(self) -> List[Optional[torch.Tensor]]:
        leading_shape = self.array.shape[:-1]
        if self.zero_flags is None:
            zeros = [False] * len(self.irreps)
        else:
            zeros = self.zero_flags

        if len(self.irreps) == 1:
            mul, ir = self.irreps[0]
            if zeros[0]:
                return [None]
            return [torch.reshape(self.array, leading_shape + (mul, ir.dim))]
        else:
            return [
                None
                if zero
                else torch.reshape(self.array[..., i], leading_shape + (mul, ir.dim))
                for zero, i, (mul, ir) in zip(zeros, self.irreps.slices(), self.irreps)
            ]
    
    @property
    def shape(self):
        r"""Shape. Equivalent to ``self.array.shape``."""
        return self.array.shape
    
    @property
    def dtype(self):
        r"""dtype. Equivalent to ``self.array.dtype``."""
        return self.array.dtype
    
    @property
    def ndim(self):
        r"""Number of dimensions. Equivalent to ``self.array.ndim``."""
        return len(self.shape)

    def __len__(self):  # noqa: D105
        return len(self.array)






irreps = o3.Irreps("0e + 1e")

input1 = torch.ones(4).to(device='cuda')
input2 = torch.ones(4).to(device='cuda')

x = IrrepsArray(input1, irreps)
y = IrrepsArray(input2, irreps)

print(x.irreps)
print(x.chunks)

def outer_product(x, y):
    return torch.einsum("...i, ...j -> ...ij", x.array, y.array)

print_performance(lambda: outer_product(x, y))

compile_sum = torch.compile(outer_product, fullgraph=True)

print_performance(lambda: compile_sum(x, y))

@torch.compile(fullgraph=True)
def outer_product_irrepsarray(x, y):
    return IrrepsArray(
        torch.einsum("...i, ...j -> ...ij", x.array, y.array),
        irreps
    )

print_performance(lambda: outer_product_irrepsarray(x, y))
