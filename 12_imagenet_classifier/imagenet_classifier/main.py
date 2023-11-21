import wandb
import argparse
import torch
import torchinfo
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

from imagenet_classifier import data_setup, engine, utils


def parse_arguments() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "Default run", help = "Name of experiment")
    parser.add_argument("--data_dir", type = str, default = "/mnt/e/Datasets/ImageNet/", help = "Path to data directory")
    parser.add_argument("--download_dir", type = str, default = "models/", help = "Path to download directory")
    parser.add_argument("--results_path", type = str, default = "results/", help = "Path to save results")
    parser.add_argument("--model_name", type = str, default = "CNN", help = "Name of model to train")
    parser.add_argument("--epochs", type = int, default = 5, help = "Number of epochs to train the model for")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Number of samples per batch")
    parser.add_argument("--lr", type = float, default = 0.1, help = "Learning rate for optimizer")
    parser.add_argument("--num_workers", type = int, default = 1, help = "Number of workers for DataLoader")
    parser.add_argument("--resize", type = tuple, default = (64, 64), help = "Resize size for images")
    parser.add_argument("--device", type = torch.device, default = "cuda" if torch.cuda.is_available() else "cpu", help = "Device to train model on")
    parser.add_argument("--verbose", type = bool, default = True, help = "Whether or not to print results")
    return parser.parse_args()


def print_model_info(model: torch.nn.Module) -> None:
    """Prints model information.

    Prints model information using torchinfo.summary().
    """
    print("\nModel information:")
    torchinfo.summary(
        model = model,
        input_size = (32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names = ["input_size"], # uncomment for smaller output
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        col_width = 20,
        row_settings = ["var_names"]
    )
    print("\n\n")


if __name__ == "__main__":
    # Initialize constants
    args = parse_arguments()
    EXPERIMENT_NAME = args.experiment_name
    DATA_DIR = Path(args.data_dir)
    DOWNLOAD_DIR = Path(args.download_dir)
    RESULTS_PATH = Path(args.results_path)
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_WORKERS = args.num_workers
    RESIZE = args.resize
    DEVICE = args.device
    VERBOSE = args.verbose

    # Create dataloaders
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
        ds_path = DATA_DIR,
        transform = transform,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    exit()

    # Create teacher and student models
    teacher_model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )
    student_model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )

    # Freeze all layers to make parameters untrainable
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in student_model.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer with a new one that has 10 output classes
    teacher_model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 10)
    )
    student_model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 10)
    )
    print_model_info(teacher_model)

    # Create optimizer and loss function
    teacher_optimizer = torch.optim.SGD(
        params = teacher_model.parameters(),
        lr = LR
    )
    student_optimizer = torch.optim.SGD(
        params = student_model.parameters(),
        lr = LR
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create model, loss function and optimizer for training
    mod_teach_results = engine.train(
        model = teacher_model,
        model_name = "ResNet50_teacher",
        train_dataloader = train_lab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = teacher_optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE
    )

    # Create pseudo-labels
    pseudo_labels = engine.get_pseudo_labels(
        model = teacher_model,
        data_loader = train_unlab_dataloader,
        device = DEVICE
    )

    # Train student model
    model_stud_results = engine.train(
        model = student_model,
        model_name = "ResNet50_student",
        train_dataloader = train_unlab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = student_optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE,
        pseudo_labels = pseudo_labels
    )