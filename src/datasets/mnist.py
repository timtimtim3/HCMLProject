import torch
from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST(MNIST):
    """
    Wrapper of MNIST from pytorch
    """

    NUM_CLASSES = 10
    
    def __init__(self, split = "train", transform = [], label_noise=0.0, force_download = False):

        # Overwrite rootdirectory
        root = "data/"
        train = True if split == "train" else False

        _transform = transforms.Compose([
            transforms.ToTensor(),
        ] + transform)

        super().__init__(root, train, _transform, None, download=True)

        
        # TODO
        # Find how labels are stored
        # Change labels, indicated by label_noise
        # Use utils/functions -> add_label_noise(.., ..., self.NUM_CLASSES)

        # Make your life easier by running the script from the src folder
        # e.g. test_script.py