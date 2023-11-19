from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List
import os
import zipfile
import requests


NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        ds_path: Path,
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
    ds_path.mkdir(parents = True, exist_ok = True)
    train_dataset = datasets.CIFAR10(root = ds_path,
                                     train = True,
                                     transform = transform,
                                     download = True)
    test_dataset = datasets.CIFAR10(root = ds_path,
                                    train = False,
                                    transform = transform,
                                    download = True)
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
    DS_PATH = Path("/mnt/e/Datasets/CIFAR10/")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader, classes = create_dataloaders(
        ds_path = DS_PATH,
        transform = transform,
        batch_size = 64,
        num_workers = os.cpu_count(),
        shuffle = True
    )
    X_train, y_train = next(iter(train_dataloader))
    print(classes)
    print(X_train.shape, y_train.shape)