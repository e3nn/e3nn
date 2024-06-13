from typing import NamedTuple, Optional


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_slice: Chunk
    input_2_slice: Chunk
    output_slice: Chunk
