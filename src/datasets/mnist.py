import torch
from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST(MNIST):
    """
    Wrapper of MNIST from pytorch
    """

    NUM_CLASSES = 10
    
    def __init__(self, split = "train", transform = [], force_download = False):

        # Overwrite rootdirectory
        root = "data/"
        train = True if split == "train" else False

        _transform = transforms.Compose([
            transforms.ToTensor(),
        ] + transform)

        super().__init__(root, train, _transform, None, download=True)