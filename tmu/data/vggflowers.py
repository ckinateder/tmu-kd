import kagglehub
import numpy as np
from tmu.data import TMUDataset
import pathlib, os
import random
import requests
import pdb
from PIL import Image
from tqdm import tqdm, trange
random.seed(42)
from tmu.util import train_test_split

class VGGFlowers(TMUDataset):
    def __init__(self, size: int = 64, tt_split: float = 0.8):
        self.size = size # size of the images to be resized to
        self.tt_split = tt_split
        super().__init__()

    def _retrieve_dataset(self) -> dict:
        # Download latest version
        path = kagglehub.dataset_download("arjun2000ashok/vggflowers")
        path = os.path.join(path, "images")
        files_and_labels = requests.get("https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/refs/heads/master/filelists/flowers/base.json").json()
        images_names = files_and_labels["image_names"]
        labels = files_and_labels["image_labels"]
        images = []

        for i in trange(len(images_names), leave=False, desc="resizing images"):
            images_names[i] = images_names[i].replace("filelists/flowers/images/", "")
            images_names[i] = os.path.join(path, images_names[i])
            # ensure that the image exists
            assert os.path.exists(images_names[i])

            # open the image
            image = Image.open(images_names[i])
            image = image.resize((self.size, self.size))
            image = np.array(image)
            images.append(image)

        assert len(images) == len(labels) == len(images_names)
        images = np.array(images)
        labels = np.array(labels)

        # split the dataset
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=self.tt_split, random_state=42)

        return dict(
            x_train=images_train,
            y_train=labels_train,
            x_test=images_test,
            y_test=labels_test
        )
        
        

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset.astype(np.uint32)

        return dataset.astype(np.uint32)
    

if __name__ == "__main__":
    size = 64
    tt_spilt = .8

    vggflowers_ds = VGGFlowers(size=size, tt_split=tt_spilt)
    data = vggflowers_ds.get()
    info = f"{size}x{size}_{int(tt_spilt*100)}"
    print(f"Saving as vggflowers_{info}.npy")

    # save as npy files
    np.save(f"vggflowers_x_train_{info}.npy", data["x_train"])
    np.save(f"vggflowers_y_train_{info}.npy", data["y_train"])
    np.save(f"vggflowers_x_test_{info}.npy", data["x_test"])
    np.save(f"vggflowers_y_test_{info}.npy", data["y_test"])
    print("Done")

