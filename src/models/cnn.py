import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dataclasses import SampleInfo


class CNN(nn.Module):
    """
    A customizable Convolutional Neural Network (CNN) for image classification tasks.

    Args:
        input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB images).
        output_size (int): Number of output classes.
    """

    def __init__(self, sample_info: SampleInfo, hidden_sizes):
        super().__init__()

        # Create two convolution layers with pooling after each layer
        self.conv1 = nn.Conv2d(sample_info.input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # Create fully connected layers
        img_w_and_h = sample_info.input_size

        # Size after two convolutions and two pooling operations
        in_size = int(((((img_w_and_h - 4) / 2) - 4) / 2)**2 * 16)
        layers = []

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, sample_info.output_size))

        # Combine layers into a Sequential model
        self.fc = nn.Sequential(*layers)


    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)

        return x
