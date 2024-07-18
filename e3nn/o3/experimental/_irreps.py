# The existing Irreps class does not play nice with torch.compile. Sorry !

import torch
from typing import List, Tuple, Union

class Irrep:
    def __init__(self, l: int, p: int):
        self.l = l
        self.p = p
    
    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    def __repr__(self) -> str:
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

class Irreps:
    def __init__(self, irreps: List[Tuple[int, Irrep]]):
        self.irreps = irreps
    
    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self.irreps)

    def __repr__(self) -> str:
        return "+".join(f"{mul}x{ir}" for mul, ir in self.irreps)

    @staticmethod
    def parse(irreps_str: str) -> 'Irreps':
        parts = irreps_str.split("+")
        irreps = []
        for part in parts:
            if "x" in part:
                mul, ir = part.split("x")
                mul = int(mul)
            else:
                mul = 1
                ir = part
            l = int(ir.strip()[:-1])
            p = 1 if ir[-1] == "e" else -1
            irreps.append((mul, Irrep(l, p)))
        return Irreps(irreps)

    def randn(self, *size: int, normalization: str = "component", 
              requires_grad: bool = False, dtype=None, device=None) -> torch.Tensor:
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]
        
        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            start = 0
            for mul, ir in self.irreps:
                end = start + mul * ir.dim
                r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                r.div_(r.norm(2, dim=di + 1, keepdim=True))
                x.narrow(di, start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
                start = end
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")