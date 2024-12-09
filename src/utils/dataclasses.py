from dataclasses import dataclass


@dataclass
class SampleInfo:
    input_size: int
    output_size: int
    input_channels: int
    flatten_input_size: int


@dataclass
class ModelCheckpoint:
    epoch: int
