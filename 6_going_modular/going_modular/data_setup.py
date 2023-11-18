from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List
import os


def create_dataloaders(
        train_dir: str | Path,
        test_dir: str,
        transform: transforms,
        batch_size: int,
        num_workers: int = os.cpu_count(),
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
        num_workers = 4
    )
    print(classes)