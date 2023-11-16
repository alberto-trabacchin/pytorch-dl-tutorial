import torch
import torchvision

if __name__ == "__main__":
    train_dataset = torchvision.datasets.Food101(
        root = "datasets/Food101/",
        download = True,
        transform = torchvision.transforms.ToTensor(),
        split = "train"
    )