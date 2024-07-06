from typing import NamedTuple, Optional, Tuple
from dataclasses import dataclass, field, replace
import torch


class Chunk(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Path(NamedTuple):
    input_1_chunk: Chunk
    input_2_chunk: Chunk
    output_chunk: Chunk


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Instruction:
    """Defines an instruction for a tensor product."""

    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    weight_std: float
    path_shape: Tuple[int, ...] = field(init=False)
    chunk_in1: Optional[Chunk] = None
    chunk_in2: Optional[Chunk] = None
    chunk_out: Optional[Chunk] = None
    num_elements: int = field(init=False)

    def __post_init__(self):
        if self.connection_mode not in [
            "uvw",
            "uvu",
            "uvv",
            "uuw",
            "uuu",
            "uvuv",
            "uvu<v",
            "u<vw",
        ]:
            raise ValueError(
                f"Unsupported connection_mode {self.connection_mode} for instruction."
            )

        path_shape = {
            "uvw": (
                self.chunk_in1.mul,
                self.chunk_in2.mul,
                self.chunk_out.mul,
            ),
            "uvu": (self.chunk_in1.mul, self.chunk_in2.mul),
            "uvv": (self.chunk_in1.mul, self.chunk_in2.mul),
            "uuw": (self.chunk_in1.mul, self.chunk_out.mul),
            "uuu": (self.chunk_in1.mul,),
            "uvuv": (self.chunk_in1.mul, self.chunk_in2.mul),
            "uvu<v": (
                self.chunk_in1.mul
                * (self.chunk_in2.mul - 1)
                // 2,
            ),
            "u<vw": (
                self.chunk_in1.mul
                * (self.chunk_in2.mul - 1)
                // 2,
                self.chunk_out.mul,
            ),
        }[self.connection_mode]
        super().__setattr__("path_shape", path_shape)

        num_elements = {
            "uvw": (self.chunk_in1.mul * self.chunk_in2.mul),
            "uvu": self.chunk_in2.mul,
            "uvv": self.chunk_in1.mul,
            "uuw": self.chunk_in1.mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.chunk_in1.mul
            * (self.chunk_in2.mul - 1)
            // 2,
        }[self.connection_mode]
        super().__setattr__("num_elements", num_elements)

    def replace(self, **changes) -> "Instruction":
        return replace(self, **changes)

    def __repr__(self) -> str:
        return (
            "Instruction("
            + ", ".join(
                [
                    f"i={self.i_in1},{self.i_in2},{self.i_out}",
                    f"mode={self.connection_mode}",
                    f"has_weight={self.has_weight}",
                    f"path_weight={self.path_weight}",
                    f"weight_std={self.weight_std}",
                    f"mul={self.chunk_in1.mul},{self.chunk_in2.mul},{self.chunk_out.mul}",
                    f"path_shape={self.path_shape}",
                    f"num_elements={self.num_elements}",
                ]
            )
            + ")"
        )