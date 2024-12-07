from torch.utils.data import Dataset
import subprocess
import os
import zipfile
import h5py
from PIL import Image
import io

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

        self._download(force_download)
        # self._preprocess()
        # self._save()

        self._load()

        # # Don't forget to use the transform list
        # transform_data = transforms.Compose([
        #     transforms.ToTensor(),
        # ] + transform)

    def _run_kaggle_download(self, dataset_slug, filename, force_download=False):
        """
        A helper function to download a specific file from a Kaggle dataset, 
        unzip it, and remove the zip file.

        Args:
            dataset_slug (str): The Kaggle dataset identifier, e.g. "username/datasetname".
            filename (str): The specific file to download (e.g. "image_256sq.hdf5").
            force_download (bool): If True, re-download even if the file exists.
        """
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.isfile(file_path) or force_download:
            # Download the file
            print(f"Downloading {filename}...")
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", dataset_slug,
                "-f", filename,
                "-p", OUTPUT_DIR
            ], check=True)

            zip_path = file_path + ".zip"
            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)

            # Remove the zip file
            os.remove(zip_path)
            print(f"{filename} download and extraction complete.")
        else:
            print(f"{filename} already exists, skipping download.")

    def _download(self, force_download=False):
        # Ensure OUTPUT_DIR exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        dataset_slug = "tomooinubushi/all-isic-data-20240629"

        # Download HDF5
        self._run_kaggle_download(dataset_slug, "image_256sq.hdf5", force_download)

        # Download Metadata CSV
        self._run_kaggle_download(dataset_slug, "metadata.csv", force_download)

    def _load(self):
        hdf5_path = os.path.join(self.root, "image_256sq.hdf5")
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"{hdf5_path} not found. Make sure the file exists or run download first.")

        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            # print(len(list(f.keys())))
            binary_data = f['ISIC_9965754'][()]
            img = Image.open(io.BytesIO(binary_data))

            print(img)  # This will tell you what type of object it is            
            # Adjust the dataset names to match what’s inside your HDF5 file.
            # For example, if your file has a dataset called "images" and one called "labels":
            # self.images = f["images"][:]   # Reads the entire dataset into memory as a NumPy array
            # self.labels = f["labels"][:]   # Same for labels

    def _preprocess(self):
        # images need to be (input_channels, height, width)
        pass
    
    def _save(self):
        pass


dataset = ISIC2024("hi")
