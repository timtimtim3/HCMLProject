import os
from torch.utils.data import Dataset
from torchvision import transforms


TRAIN_DOWLOAD_URL = ""
TEST_DOWNLOAD_URL = ""
OUTPUT_DIR = "data/ISIC2020"


class ISIC2020(Dataset):

    def __init__(self, split, transform=[], download=True):

        self.root = "data/ISIC2020"
        self.split = split  # can only be "train" or "val"

        # Is used as input for MLP
        self.data_dim = ...

        # Used for model output
        self.label_dim = ...

        # Used for first layer in CNN
        self.input_channels = ...

        if download:
            self._download()
            self._preprocess()
            self._save()


        # Don't forget to use the transform list
        transform_data = transforms.Compose([
            transforms.ToTensor(),
        ] + transform)


    def _download(self):
        # create OUTPUT_DIR
        # os.system(f"wget {TRAIN_DOWLOAD_URL} -o {OUTPUT_DIR}")
        pass


    def _preprocess(self):
        # images need to be (input_channels, height, width)
        pass

    
    def _save(self):
        pass
    