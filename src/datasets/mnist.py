import os
import pickle
import numpy as np
import torch
from torchvision.datasets import MNIST as TorchvisionMNIST
from torchvision import transforms
from utils.functions import add_label_noise

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

class MNIST(TorchvisionMNIST):
    """
    Wrapper of the MNIST dataset from pytorch.
    """

    NUM_CLASSES = 10

    def __init__(self, split="train", transform=[], label_noise=0.0, seed=42, skip_indices=[]):
        self.root = OUTPUT_DIR

        # Determine if we are in train mode or not
        train = (split == "train")

        # Compose the transforms
        _transform = transforms.Compose([
                                            transforms.ToTensor(),
                                        ] + transform)

        super().__init__(self.root, train=train, transform=_transform, download=True)

        self.split = split
        self.label_noise = label_noise

        # If this is the training split and we have a non-zero noise level, add noise
        if self.split == 'train' and self.label_noise > 0:
            os.makedirs(self.root, exist_ok=True)

            # Convert targets to numpy array for convenience
            labels = np.array(self.targets)

            # Save original labels
            np.save(os.path.join(self.root, "MNIST", "labels_train.npy"), labels)

            # Generate noisy labels
            noisy_labels = add_label_noise(labels, noise_level=self.label_noise, num_classes=self.NUM_CLASSES)

            # Assign noisy labels back to the dataset
            self.targets = torch.tensor(noisy_labels, dtype=torch.long)

            # Save the noisy labels
            np.save(os.path.join(self.root, "MNIST", f"labels_train_noisy{self.label_noise}.npy"), noisy_labels)


        if len(skip_indices) != 0:
            self.data = [d for i,d in enumerate(self.data) if i not in skip_indices]
            self.targets = [t for i,t in enumerate(self.targets) if i not in skip_indices]
