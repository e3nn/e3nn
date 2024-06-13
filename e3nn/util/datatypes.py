from typing import NamedTuple, Optional


class Path(NamedTuple):
    mul: int
    dim: int
    slice: Optional[slice] = None


class Paths(NamedTuple):
    input_1_slice: Path
    input_2_slice: Path
    output_slice: Path
