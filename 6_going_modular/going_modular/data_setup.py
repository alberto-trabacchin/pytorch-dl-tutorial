from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List
import os
import zipfile
import requests


NUM_WORKERS = os.cpu_count()


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
        
        with zipfile.ZipFile(download_path / "pizza_steak_sushi.zip", "r") as zip_f:
            print("Unzipping dataset to ", dataset_path)
            zip_f.extractall(dataset_path)


def create_dataloaders(
        train_dir: str | Path,
        test_dir: str,
        transform: transforms,
        batch_size: int,
        num_workers: int = NUM_WORKERS,
        shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        shuffle: Whether or not to shuffle the data in the DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir = path/to/train_dir,
                                test_dir = path/to/test_dir,
                                transform = some_transform,
                                batch_size = 32,
                                num_workers = 4)
    """
    train_dataset = datasets.ImageFolder(root = train_dir, 
                                         transform = transform)
    test_dataset = datasets.ImageFolder(root = test_dir,
                                        transform = transform)
    class_names = train_dataset.classes
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = shuffle,
                                  num_workers = num_workers)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size = batch_size,
                                 shuffle = shuffle,
                                 num_workers = num_workers)
    return train_dataloader, test_dataloader, class_names


if __name__ == "__main__":
    DS_PATH = Path("datasets/pizza_steak_sushi")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader, classes = create_dataloaders(
        train_dir = DS_PATH / "train",
        test_dir = DS_PATH / "test",
        transform = transform,
        batch_size = 64,
        num_workers = os.cpu_count(),
        shuffle = True
    )
    X_train, y_train = next(iter(train_dataloader))
    print(classes)
    print(X_train.shape, y_train.shape)