from typing import NamedTuple, Optional
import torch


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_chunk: Chunk
    input_2_chunk: Chunk
    output_chunk: Chunk


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float
    chunk_in: Optional[Chunk] = None
    chunk_out: Optional[Chunk] = None
