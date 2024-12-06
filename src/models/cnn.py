import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    A customizable Convolutional Neural Network (CNN) for image classification tasks.

    Args:
        input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB images).
        num_classes (int): Number of output classes.
    """

    def __init__(self, input_size, hidden_sizes, output_size, input_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)


    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # import pdb; pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
