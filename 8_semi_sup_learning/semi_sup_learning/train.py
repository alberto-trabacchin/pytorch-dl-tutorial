import argparse
import torch
from torchvision.transforms import transforms
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
import torchinfo

from semi_sup_learning import data_setup, engine, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "/mnt/d/Datasets/CIFAR-10/", help = "Path to data directory")
    parser.add_argument("--download_dir", type = str, default = "models/", help = "Path to download directory")
    parser.add_argument("--model_name", type = str, default = "TinyVGG", help = "Name of model to train")
    parser.add_argument("--input_size", type = int, default = 3, help = "Number of input channels")
    parser.add_argument("--hidden_size", type = int, default = 5, help = "Number of hidden units")
    parser.add_argument("--output_size", type = int, default = 3, help = "Number of output units")
    parser.add_argument("--model_dir", type = str, default = "models/", help = "Path to save trained model")
    parser.add_argument("--model_save", type = bool, default = False, help = "Whether or not to save trained model")
    parser.add_argument("--epochs", type = int, default = 5, help = "Number of epochs to train the model for")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Number of samples per batch")
    parser.add_argument("--lr", type = float, default = 0.1, help = "Learning rate for optimizer")
    parser.add_argument("--num_workers", type = int, default = 1, help = "Number of workers for DataLoader")
    parser.add_argument("--resize", type = tuple, default = (64, 64), help = "Resize size for images")
    parser.add_argument("--verbose", type = bool, default = True, help = "Whether or not to print results")
    parser.add_argument("--device", type = torch.device, default = "cuda" if torch.cuda.is_available() else "cpu", help = "Device to train model on")
    parser.add_argument("--labels_ratio", type = float, default = 0.3, help = "Ratio of labeled data on the training dataset for training the teacher")
    parser.add_argument("--results_path", type = str, default = "results/", help = "Path to save results")

    args = parser.parse_args()

    # Initialize constants
    DS_PATH = Path(args.data_dir)
    MODEL_NAME = args.model_name
    MODEL_DIR = Path(args.model_dir)
    MODEL_SAVE = args.model_save
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_WORKERS = args.num_workers
    RESIZE = args.resize
    VERBOSE = args.verbose
    DEVICE = args.device
    LABELS_RATIO = args.labels_ratio
    RESULTS_PATH = Path(args.results_path)

    # Create dataloaders
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor()
    ])
    train_lab_dataloader, train_unlab_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
        ds_path = DS_PATH,
        labels_ratio = LABELS_RATIO,
        transform = transform,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )
    print(f"Length of labeled training dataset: {len(train_lab_dataloader.dataset)}")
    print(f"Length of unlabeled training dataset: {len(train_unlab_dataloader.dataset)}")
    print(f"Length of test dataset: {len(test_dataloader.dataset)}")

    # Create teacher and student models
    teacher_model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )
    student_model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in student_model.parameters():
        param.requires_grad = False
    teacher_model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 10)
    )
    student_model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 10)
    )

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
        model_name = "resnet50_teacher",
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
        model_name = "resnet50_student",
        train_dataloader = train_unlab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = student_optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE,
        pseudo_labels = pseudo_labels
    )
    utils.save_model(
        name = f"resnet50_teacher_{EPOCHS}_epochs_{LABELS_RATIO}_labels_ratio.pth",
        model = teacher_model,
        model_dir = MODEL_DIR
    )
    utils.save_model(
        name = f"resnet50_student_{EPOCHS}_epochs_{LABELS_RATIO}_labels_ratio.pth",
        model = student_model,
        model_dir = MODEL_DIR
    )
    utils.plot_accuracies(
        model_results = mod_teach_results,
        save_path = RESULTS_PATH / f"resnet50_teacher_{EPOCHS}_epochs_{LABELS_RATIO}_labels_ratio.png"
    )
    utils.plot_accuracies(
        model_results = model_stud_results,
        save_path = RESULTS_PATH / f"resnet50_student_{EPOCHS}_epochs_{LABELS_RATIO}_labels_ratio.png"
    )