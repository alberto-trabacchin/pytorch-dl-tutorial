import wandb
import argparse
import torch
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = Path("/mnt/d/Datasets/CIFAR-10/"), help = "Path to data directory")
    parser.add_argument("--download_dir", type = str, default = Path("models/"), help = "Path to download directory")
    parser.add_argument("--results_path", type = str, default = Path("results/"), help = "Path to save results")
    parser.add_argument("--model_name", type = str, default = "CNN", help = "Name of model to train")
    parser.add_argument("--epochs", type = int, default = 5, help = "Number of epochs to train the model for")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Number of samples per batch")
    parser.add_argument("--lr", type = float, default = 0.1, help = "Learning rate for optimizer")
    parser.add_argument("--num_workers", type = int, default = 1, help = "Number of workers for DataLoader")
    parser.add_argument("--resize", type = tuple, default = (64, 64), help = "Resize size for images")
    parser.add_argument("--device", type = torch.device, default = "cuda" if torch.cuda.is_available() else "cpu", help = "Device to train model on")
    parser.add_argument("--labels_ratio", type = float, default = 0.3, help = "Ratio of labeled data on the training dataset for training the teacher")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project-2",
        name = "Test run",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "epochs": 100,
        }
    )