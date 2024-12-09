from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class SampleInfo:
    input_size: int
    output_size: int
    input_channels: int
    flatten_input_size: int


@dataclass
class ModelCheckpoint:
    epoch: int
    model_state_dict: OrderedDict
    optimizer_state_dict: dict
    learning_rate: float
    training_loss: float
    validation_loss: float
    validation_accuracy: float
    precision_recall_f1: tuple[float]
