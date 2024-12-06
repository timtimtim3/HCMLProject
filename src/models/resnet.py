import torch.nn as nn
from torchvision.models import resnet18 as rs18, resnet34 as rs34, resnet50 as rs50,\
      ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision import transforms

# NOTE: Every ResNet has an input_size of 256 and needs a dataset which is pre-processed 
# (see: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)


def get_resnet_transform():

    return [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def ResNet18(input_size, hidden_sizes, output_size, input_channels):
    return build_resnet(rs18, ResNet18_Weights, output_size)


def ResNet34(input_size, hidden_sizes, output_size, input_channels):
    return build_resnet(rs34, ResNet34_Weights, output_size)


def ResNet50(input_size, hidden_sizes, output_size, input_channels):
    return build_resnet(rs50, ResNet50_Weights, output_size)


def build_resnet(template, weights, output_size):

    model = template(weights)
    model.fc = nn.Linear(512, output_size)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model