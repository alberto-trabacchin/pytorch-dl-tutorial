import torch
import torchvision
from pathlib import Path
import requests
import zipfile

def download_dataset(download_path, dataset_path):
    if dataset_path.is_dir():
        print("Dataset path already exists: ", dataset_path)
    else:
        print("Dataset path does not exist")
        dataset_path.mkdir(parents = True, exist_ok = True)

        with open(download_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi dataset as pizza_steak_sushi.zip...")
            f.write(request.content)
        
        with zipfile.ZipFile(dataset_path / "pizza_steak_sushi.zip", "r") as zip_f:
            print("Unzipping dataset to ", dataset_path)
            zip_f.extractall(dataset_path)


if __name__ == "__main__":
    DS_DOWNL_PATH = Path("datasets/")
    DS_PATH = DS_DOWNL_PATH / "pizza_steak_sushi"
    download_dataset(DS_DOWNL_PATH, DS_PATH)
    