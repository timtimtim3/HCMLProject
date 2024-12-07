from torch.utils.data import Dataset
import subprocess
import os
import zipfile


OUTPUT_DIR = "data/ISIC2024"


class ISIC2024(Dataset):

    def __init__(self, split, transform=[], force_download=False):

        self.root = OUTPUT_DIR
        self.split = split  # can only be "train" or "val"

        # Is used as input for MLP
        self.data_dim = ...

        # Used for model output
        self.label_dim = ...

        # Used for first layer in CNN
        self.input_channels = ...

        self._download(force_download=force_download)
        # self._preprocess()
        # self._save()


        # # Don't forget to use the transform list
        # transform_data = transforms.Compose([
        #     transforms.ToTensor(),
        # ] + transform)

    def _download(self, force_download=False):
        hdf5_path = os.path.join(OUTPUT_DIR, "image_256sq.hdf5")
        if not os.path.isfile(hdf5_path) or force_download:
            # Ensure OUTPUT_DIR exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Download the file
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", "tomooinubushi/all-isic-data-20240629", 
                "-f", "image_256sq.hdf5", 
                "-p", OUTPUT_DIR
            ], check=True)

            zip_path = os.path.join(OUTPUT_DIR, "image_256sq.hdf5.zip")

            # Unzip using Python's zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)

            # Remove the zip file
            os.remove(zip_path)
        else:
            print("File already exists, skipping download.")

    def _preprocess(self):
        # images need to be (input_channels, height, width)
        pass
    
    def _save(self):
        pass


# dataset = ISIC2024("hi")
