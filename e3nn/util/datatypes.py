from typing import NamedTuple, Optional
from enum import Enum, auto

class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None

class TensorProductMode(Enum):
    UUU = auto()
    UVUV = auto()
    UVW = auto()

class Path(NamedTuple):
    input_1_slice: Chunk
    input_2_slice: Optional[Chunk] = None
    output_slice: Optional[Chunk] = None
    tensor_product_mode: TensorProductMode = TensorProductMode.UUU
