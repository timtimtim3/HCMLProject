from torchvision.datasets import MNIST
from torchvision import transforms


class MNIST(MNIST):
    """
    Wrapper of MNIST from pytorch
    """
    
    def __init__(self, split = "train", transform = [], download = False):

        if split == "train":
            train = True
        else:
            train = False

        self.label_dim = 10
        self.data_dim = 784
        self.input_channels = 1

        _transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.unsqueeze(1))  # add 1 dim for color channel
        ] + transform)

        # Overwrite rootdirectory
        root = "data/"

        super().__init__(root, train, _transform, None, download)