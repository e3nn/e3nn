from typing import NamedTuple, Optional


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_chunk: Chunk
    input_2_chunk: Chunk
    output_chunk: Chunk
