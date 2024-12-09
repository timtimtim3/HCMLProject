import torch
import torch.nn as nn

from utils.dataclasses import SampleInfo


class MLP(nn.Module):

    def __init__(self, sample_info: SampleInfo, hidden_sizes):
        super().__init__()

        layers = []
        in_size = sample_info.flatten_input_size

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
        x = torch.flatten(x, start_dim=1)

        return self.fc(x)