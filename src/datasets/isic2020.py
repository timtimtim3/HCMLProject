import os 

TRAIN_DOWLOAD_URL = ""
TEST_DOWNLOAD_URL = ""


class ISIC:

    def __init__(self, split: "train" | "test"):

        self.split = split
        pass


    def _download(self):
        # Run
        os.system(f"wget {TRAIN_DOWLOAD_URL}")

dataset = ISIC("test")