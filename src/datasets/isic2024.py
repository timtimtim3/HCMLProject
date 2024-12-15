import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import subprocess
import os
import zipfile
import h5py
from PIL import Image
import io
import pandas as pd
from utils.functions import map_isic_label_to_binary, add_label_noise
from torchvision import transforms

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ISIC2024')


class ISIC2024(Dataset):
    
    NUM_CLASSES = 2

    def __init__(self, split, transform=None, force_download=False, train_ratio=0.8, val_ratio=0.1, label_noise=0.0,
                 seed=42, skip_indices=[]):
        """
        Args:
            split (str): "train", "val", or "test"
            transform (list): Optional transforms to apply.
            force_download (bool): If True, force re-download of data.
            train_ratio (float): Ratio for training split. E.g., 0.8 means 80% of data is train.
            val_ratio (float): Ratio for validation split.
        """
        self.seed = seed
        self.root = OUTPUT_DIR
        self.split = split  # "train", "val", or "test"
        assert self.split in ["train", "val", "test"], "split must be one of train, val, test"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.noise_level = label_noise

        if transform is None:
            transform = []

        # Download data if needed
        self._download(force_download)

        # Load metadata and filter images
        self._load_metadata()

        # Load keys from HDF5 and match them to labels
        self._load()

        # Split data
        self._split(split=split)

        if split == 'train' and label_noise > 0:
            self._add_noise_and_save_labels()

        if len(skip_indices) != 0:
            skip_image_ids = [value for i, value in enumerate(self.image_ids) if i in skip_indices]
            self.image_ids = [img_id for img_id in self.image_ids if img_id not in skip_image_ids]
            self.id_to_label = {key: value for key, value in self.id_to_label.items() if key not in skip_image_ids}


        # Compose the transforms
        self.transform = transforms.Compose([
                                                transforms.ToTensor(),
                                            ] + transform)

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

    def _load_metadata(self):
        metadata_path = os.path.join(self.root, "metadata.csv")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"{metadata_path} not found. Run download first.")

        metadata_df = pd.read_csv(metadata_path)

        # Create a mapping from isic_id to binary label
        self.id_to_label = {}
        for idx, row in metadata_df.iterrows():
            isic_id = row["isic_id"]
            binary_label = map_isic_label_to_binary(row["benign_malignant"])
            if binary_label is not None:
                self.id_to_label[isic_id] = binary_label

    def _load(self):
        hdf5_path = os.path.join(self.root, "image_256sq.hdf5")
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"{hdf5_path} not found. Make sure the file exists or run download first.")

        # Gather valid image keys from the HDF5
        with h5py.File(hdf5_path, 'r') as f:
            all_keys = list(f.keys())  # image ids like "ISIC_9995837"
            # Filter keys to those that are in id_to_label
            self.image_ids = [k for k in all_keys if k in self.id_to_label]

        # Print how many we have after filtering
        print(f"Found {len(self.image_ids)} images with binary labels.")

        # We don't read the images into memory now; we access them on-demand in __getitem__
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')

    def _split(self, split):
        random.seed(self.seed)

        # Create a reproducible split
        random.shuffle(self.image_ids)

        total = len(self.image_ids)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        if split == 'train':
            self.image_ids = self.image_ids[:train_end]
        elif split == 'val':
            self.image_ids = self.image_ids[train_end:val_end]
        else:
            self.image_ids = self.image_ids[val_end:]

    def _add_noise_and_save_labels(self):
        # Use the order of self.image_ids to define the indexing
        labels = np.array([self.id_to_label[iid] for iid in self.image_ids])

        # Save the original labels in the same order as self.image_ids
        np.save(os.path.join(self.root, 'ids_train.npy'), self.image_ids)
        np.save(os.path.join(self.root, 'labels_train.npy'), labels)

        # Apply noise
        noisy_labels = add_label_noise(labels, noise_level=self.noise_level, num_classes=self.NUM_CLASSES)

        self.id_to_label = dict(zip(self.image_ids, noisy_labels))

        # Save the noisy labels aligned with the original indexing order
        np.save(os.path.join(self.root, f'labels_train_noisy{self.noise_level}.npy'), noisy_labels)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.id_to_label[image_id]

        binary_data = self.hdf5_file[image_id][()]

        # Convert binary JPEG to PIL Image
        img = Image.open(io.BytesIO(binary_data)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            # If no transform, convert PIL to tensor anyway
            img = transforms.ToTensor()(img)

        return img, label
