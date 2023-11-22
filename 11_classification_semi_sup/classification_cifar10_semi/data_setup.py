from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
from typing import Tuple, List
import os
import zipfile
import requests


NUM_WORKERS = os.cpu_count()

class PseudoDataset(Subset):
    def __init__(self, dataset: Subset):
        super().__init__(dataset, dataset.indices)
        self.pseudo_labels = {"idx", "label"}

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return (self.dataset[[self.indices[i] for i in idx]], self.pseudo_labels)
        return (self.dataset[self.indices[idx]], self.pseudo_labels[idx])
    
    def __len__(self):
        return super().__len__()
    
    def set_pseudo_labels(self, pseudo_labels):
        self.pseudo_labels = pseudo_labels


def create_dataloaders(
        ds_path: Path,
        transform: transforms,
        labels_ratio: float,
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
    train_lab_dataset, train_unlab_dataset = random_split(train_dataset, 
                                                          lengths = [labels_ratio, (1 - labels_ratio)])
    class_names = train_dataset.classes
    train_lab_dataloader = DataLoader(dataset = train_lab_dataset,
                                      batch_size = batch_size,
                                      shuffle = shuffle,
                                      num_workers = num_workers)
    train_unlab_dataloader = DataLoader(dataset = train_unlab_dataset,
                                        batch_size = batch_size,
                                        shuffle = shuffle,
                                        num_workers = num_workers)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size = batch_size,
                                 shuffle = shuffle,
                                 num_workers = num_workers)
    
    y_len = len(train_unlab_dataset)
    pseudo_labels = torch.randint(0, 10, (y_len,)).tolist()
    print(train_unlab_dataset.indices)
    exit()
    return train_lab_dataloader, train_unlab_dataloader, test_dataloader, class_names