import argparse
import torch
from torchvision.transforms import transforms
from pathlib import Path

from going_modular import data_setup, model_builder, engine, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "data/pizza_steak_sushi/", help = "Path to data directory")
    parser.add_argument("--download_dir", type = str, default = "data/", help = "Path to download directory")
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
    parser.add_argument("--resize", type = tuple, default = (512, 512), help = "Resize size for images")
    parser.add_argument("--verbose", type = bool, default = True, help = "Whether or not to print results")
    parser.add_argument("--device", type = torch.device, default = "cuda" if torch.cuda.is_available() else "cpu", help = "Device to train model on")
    args = parser.parse_args()

    # Initialize constants
    DS_DIR = Path(args.data_dir)
    DS_DOWNL_PATH = Path(args.download_dir)
    MODEL_NAME = args.model_name
    INPUT_SIZE = args.input_size
    HIDDEN_SIZE = args.hidden_size
    OUTPUT_SIZE = args.output_size
    MODEL_DIR = Path(args.model_dir)
    MODEL_SAVE = args.model_save
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_WORKERS = args.num_workers
    RESIZE = args.resize
    VERBOSE = args.verbose
    DEVICE = args.device

    # Create dataloaders
    torch.manual_seed(42)
    data_setup.download_dataset(DS_DOWNL_PATH, DS_DIR)
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
        train_dir = DS_DIR / "train",
        test_dir = DS_DIR / "test",
        transform = transform,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    # Create model, loss function and optimizer for training
    cnn_model = model_builder.TinyVGG(
        input_shape = INPUT_SIZE,
        hidden_units = HIDDEN_SIZE,
        output_shape = OUTPUT_SIZE,
        name = MODEL_NAME
    )
    linear_model = model_builder.LinearModel(
        input_size = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        output_size = OUTPUT_SIZE,
        name = MODEL_NAME
    )

    model = cnn_model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)

    # Train model
    engine.train(
        model = model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE
    )

    # Save model
    if MODEL_SAVE:
        utils.save_model(model = model, model_dir = MODEL_DIR)
