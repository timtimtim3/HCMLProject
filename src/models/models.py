

class MLP:
    pass


class DDX:
    pass


import torch.nn as nn

# Define the model with more than one hidden layer
class DDXModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DDXModel, self).__init__()
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
        return self.network(x)