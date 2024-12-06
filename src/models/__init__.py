

from models.cnn import CNN
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34, ResNet50


AVAILABLE_MODELS = {
    "mlp": MLP,
    "cnn": CNN,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50
}

