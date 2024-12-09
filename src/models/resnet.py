import torch.nn as nn
from torchvision.models import resnet18 as rs18, resnet34 as rs34, resnet50 as rs50,\
      ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision import transforms

from utils.dataclasses import SampleInfo

# NOTE: Every ResNet has an input_size of 256 and needs a dataset which is pre-processed 
# (see: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)


def get_resnet_transform():

    return [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def ResNet18(sample_info: SampleInfo, hidden_sizes):
    return build_resnet(rs18, ResNet18_Weights.DEFAULT, sample_info, hidden_sizes)


def ResNet34(sample_info: SampleInfo, hidden_sizes):
    return build_resnet(rs34, ResNet34_Weights.DEFAULT, sample_info, hidden_sizes)


def ResNet50(sample_info: SampleInfo, hidden_sizes):
    return build_resnet(rs50, ResNet50_Weights.DEFAULT, sample_info, hidden_sizes)


def build_resnet(template, weights, sample_info: SampleInfo, hidden_sizes):

    model = template(weights=weights)

    layers = []
    in_size = 512

    # Create hidden layers
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(nn.ReLU())
        in_size = hidden_size

    # Output layer
    layers.append(nn.Linear(in_size, sample_info.output_size))

    # Combine layers into a Sequential model
    model.fc = nn.Sequential(*layers)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model