import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import requests
import zipfile
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class TinyVGG(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TinyVGG, self).__init__()

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = input_size, 
                            out_channels = hidden_size, 
                            kernel_size = 2, 
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_size, 
                            out_channels = hidden_size, 
                            kernel_size = 2,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = hidden_size, 
                            out_channels = hidden_size, 
                            kernel_size = 2, 
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_size, 
                            out_channels = hidden_size, 
                            kernel_size = 2,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 5780, 
                            out_features = output_size),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
    

def download_dataset(download_path, dataset_path):
    if dataset_path.is_dir():
        print("Dataset path already exists at: ", dataset_path)
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


def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               data_loader: DataLoader,
               device: str = "cpu",
               train_losses: list = [],
               train_accuracies: list = []) -> None:
    model.train()
    batch_train_loss = 0
    batch_train_accuracy = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        optimizer.zero_grad()
        train_loss = loss_fn(y_pred, y)
        train_loss.backward()
        optimizer.step()
        batch_train_loss += train_loss.item()
        train_accuracy = accuracy(y_pred, y)
        batch_train_accuracy += train_accuracy
    train_losses.append(batch_train_loss / len(data_loader))
    train_accuracies.append(batch_train_accuracy / len(data_loader))


def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              data_loader: DataLoader,
              device: str = "cpu",
              test_losses: list = [],
              test_accuracies: list = []) -> None:
    model.eval()
    batch_test_loss = 0
    batch_test_accuracy = 0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            test_loss = loss_fn(y_pred, y)
            batch_test_loss += test_loss.item()
            test_accuracy = accuracy(y_pred, y)
            batch_test_accuracy += test_accuracy
        test_losses.append(batch_test_loss / len(data_loader))
        test_accuracies.append(batch_test_accuracy / len(data_loader))


def accuracy(y_pred_logit, y_true_classes):
    y_pred_class = torch.softmax(y_pred_logit, dim = 1).argmax(dim = 1)
    return (y_pred_class == y_true_classes).sum().item() / len(y_true_classes)


def plot_losses(train_losses, test_losses):
    fig, ax = plt.subplots()
    for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        ax.plot(train_loss, label = f"train loss {i + 1}")
        ax.plot(test_loss, label = f"test loss {i + 1}")
    ax.legend()
    ax.set_title("Train vs. Test Losses")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

def plot_accuracies(train_accuracies, test_accuracies):
    fig, ax = plt.subplots()
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        ax.plot(train_acc, label = f"train accuracy {i + 1}")
        ax.plot(test_acc, label = f"test accuracy {i + 1}")
    ax.legend()
    ax.set_title("Train vs. Test Accuracies")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")


if __name__ == "__main__":
    DS_DOWNL_PATH = Path("datasets/")
    DS_PATH = Path("datasets/pizza_steak_sushi")
    BATCH_SIZE = 32
    HIDDEN_SIZE = 20
    EPOCHS = 50
    LR = 0.001
    N_WORKERS = os.cpu_count()

    download_dataset(DS_DOWNL_PATH, DS_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transform1 = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    data_transform2 = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor()
    ])
    data_transform3 = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ToTensor()
    ])

    train_data1 = datasets.ImageFolder(
        root = DS_PATH / "train",
        transform = data_transform1
    )
    train_data2 = datasets.ImageFolder(
        root = DS_PATH / "train",
        transform = data_transform2
    )
    train_data3 = datasets.ImageFolder(
        root = DS_PATH / "train",
        transform = data_transform3
    )

    test_data = datasets.ImageFolder(
        root = DS_PATH / "test",
        transform = data_transform1
    )

    train_dataloader1 = DataLoader(
        dataset = train_data1,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = N_WORKERS
    )
    train_dataloader2 = DataLoader(
        dataset = train_data2,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = N_WORKERS
    )
    train_dataloader3 = DataLoader(
        dataset = train_data3,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = N_WORKERS
    )
    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = N_WORKERS
    )

    train_dataloaders = [
        train_dataloader1,
        train_dataloader2,
        train_dataloader3
    ]

    model1 = TinyVGG(
        input_size = 3,
        hidden_size = HIDDEN_SIZE,
        output_size = len(train_data1.classes)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model1.parameters(), lr = LR)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for i, train_dataloader in enumerate(train_dataloaders):
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        print(f"\nTraining with Dataloader {i + 1}\n")
        for epoch in tqdm(range(EPOCHS)):
            train_step(
                model1,
                optimizer,
                loss_fn,
                train_dataloader,
                device,
                train_loss,
                train_accuracy
            )
            test_step(
                model1,
                loss_fn,
                test_dataloader,
                device,
                test_loss,
                test_accuracy
            )
            tqdm.write(
                f"Epoch: {epoch + 1}/{EPOCHS}\n------------\n" \
                f"Train loss: {train_loss[-1] :.4f}\t|\t" \
                f"Train accuracy: {(train_accuracy[-1]) * 100 :2.2f}% \n" \
                f"Test loss: {test_loss[-1] :.4f}\t|\t" \
                f"Test accuracy: {(test_accuracy[-1]) * 100 :2.2f}%\n"
            )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    
    plot_losses(train_losses, test_losses)
    plot_accuracies(train_accuracies, test_accuracies)
    plt.show()
    print(train_accuracies)