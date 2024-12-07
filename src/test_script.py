from datasets.isic2024 import ISIC2024


# I removed the __init__.py files since they're not needed if you 
# run a python script in the src/ folder, like this file.


if __name__ == "__main__":

    # TODO: make train and val split

    train = ISIC2024("train", force_download=False)
    # test = ISIC2024("test", force_download=True)