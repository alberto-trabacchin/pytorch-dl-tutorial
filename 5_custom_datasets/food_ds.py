import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import requests
import zipfile
import random
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

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


def plot_transformed_images(dataset_path, transform, n = 3, seed = 42):
    random.seed(seed)
    image_paths = list(dataset_path.glob("*/*/*.jpg"))
    random_image_paths = random.sample(image_paths, k = n)
    for image_path in random_image_paths:
        image = Image.open(image_path)
        transformed_image = transform(image).permute(1, 2, 0)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title("Original image")
        ax[0].axis("off")
        ax[1].imshow(transformed_image)
        ax[1].set_title("Transformed image")
        ax[1].axis("off")


if __name__ == "__main__":
    DS_DOWNL_PATH = Path("datasets/")
    DS_PATH = DS_DOWNL_PATH / "pizza_steak_sushi"
    download_dataset(DS_DOWNL_PATH, DS_PATH)

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor()
    ])

    plot_transformed_images(DS_PATH, data_transform)

    # Create datasets
    train_data = datasets.ImageFolder(
        root = DS_PATH / "train",
        transform = data_transform,
        target_transform = None
    )
    test_data = datasets.ImageFolder(
        root = DS_PATH / "test",
        transform = data_transform,
        target_transform = None
    )
    class_dict = train_data.class_to_idx
    print(train_data)
    print(class_dict)
    print(train_data[0])

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset = train_data,
        batch_size = 1,
        shuffle = True,
        num_workers = os.cpu_count()
    )
    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = 1,
        shuffle = True,
        num_workers = os.cpu_count()
    )
    print(test_dataloader)
    plt.show()

    