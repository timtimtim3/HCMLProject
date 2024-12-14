from datasets.isic2024 import ISIC2024
from torch.utils.data import DataLoader
from utils.functions import set_seed, check_isic_data_integrity

if __name__ == "__main__":
    set_seed(42)

    train_dataset = ISIC2024("train", force_download=False, label_noise=0.1)
    val_dataset = ISIC2024("val", force_download=False)
    test_dataset = ISIC2024("test", force_download=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    check_isic_data_integrity(train_loader, val_loader, test_loader,
                              train_dataset, val_dataset, test_dataset)
