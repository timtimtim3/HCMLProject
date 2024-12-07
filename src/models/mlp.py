import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, input_channels):
        super().__init__()

        # Is used to check if the input is flattened
        self.flatten_dim = input_size * input_channels

        layers = []
        in_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))

        # Combine layers into a Sequential model
        self.network = nn.Sequential(*layers)


    def forward(self, x):

        if x.shape[1] != self.flatten_dim:
            x = torch.flatten(x, start_dim=1)

        return self.network(x)