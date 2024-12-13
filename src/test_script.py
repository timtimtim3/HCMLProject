from datasets.isic2024 import ISIC2024
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":

    # Create dataset objects
    train = ISIC2024("train", force_download=False, label_noise=0.1)
    val = ISIC2024("val", force_download=False)
    test = ISIC2024("test", force_download=False)

    # Create data loaders
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    # Test loading a few batches
    print("Testing loading batches...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Train batch {i}: images shape {images.shape}, labels shape {labels.shape}")
        if i == 1:
            break

    for i, (images, labels) in enumerate(val_loader):
        print(f"Val batch {i}: images shape {images.shape}, labels shape {labels.shape}")
        if i == 1:
            break

    for i, (images, labels) in enumerate(test_loader):
        print(f"Test batch {i}: images shape {images.shape}, labels shape {labels.shape}")
        if i == 1:
            break

    # Check that the splits contain unique image IDs and do not overlap
    # We can access image_ids from the datasets directly
    train_ids = set(train.image_ids)
    val_ids = set(val.image_ids)
    test_ids = set(test.image_ids)

    # Verify uniqueness within each split (just for sanity; typically they should already be unique)
    assert len(train_ids) == len(train.image_ids), "Duplicate IDs found in train set!"
    assert len(val_ids) == len(val.image_ids), "Duplicate IDs found in val set!"
    assert len(test_ids) == len(test.image_ids), "Duplicate IDs found in test set!"

    # Verify no overlap between sets
    assert train_ids.isdisjoint(val_ids), "Overlap found between train and val sets!"
    assert train_ids.isdisjoint(test_ids), "Overlap found between train and test sets!"
    assert val_ids.isdisjoint(test_ids), "Overlap found between val and test sets!"

    print("All checks passed successfully!")
