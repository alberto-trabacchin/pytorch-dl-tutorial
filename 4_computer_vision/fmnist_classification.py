import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import datetime


class FashionMNISTModelV1(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
    

class FashionMNISTModelV2(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.ReLU()
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits


def plot_sample(X, y, classes):
    fig, ax = plt.subplots()
    ax.imshow(X.squeeze(), cmap = "gray")
    ax.set_title(classes[y])


def accuracy(y_pred_logit, y_true_classes):
    y_pred_class = torch.softmax(y_pred_logit, dim = 1).argmax(dim = 1)
    return (y_pred_class == y_true_classes).sum().item() / len(y_true_classes)


def train_step(model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  data_loader: DataLoader,
                  device: str = "cpu",
                  train_losses: list = [],
                  train_accuracies: list = []):
    model.train()
    batch_train_loss = 0
    batch_train_accuracy = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).squeeze()
        optimizer.zero_grad()
        train_loss = loss_fn(y_pred, y)
        train_loss.backward()
        optimizer.step()
        batch_train_loss += train_loss
        train_accuracy = accuracy(y_pred, y)
        batch_train_accuracy += train_accuracy
    train_losses.append(batch_train_loss / len(train_dl))
    train_accuracies.append(batch_train_accuracy / len(train_dl))


def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              data_loader: DataLoader,
              device: str = "cpu",
              test_losses: list = [],
              test_accuracies: list = []):
    model.eval()
    batch_test_loss = 0
    batch_test_accuracy = 0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            test_loss = loss_fn(y_pred, y)
            batch_test_loss += test_loss
            test_accuracy = accuracy(y_pred, y)
            batch_test_accuracy += test_accuracy
        test_losses.append(batch_test_loss / len(test_dl))
        test_accuracies.append(batch_test_accuracy / len(test_dl))


if __name__ == "__main__":
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    Path.mkdir(Path("4_computer_vision/datasets/"), exist_ok = True)
    train_ds = datasets.FashionMNIST(
        root = "4_computer_vision/datasets/",
        train = True,
        download = True,
        transform = transforms.ToTensor()
    )
    test_ds = datasets.FashionMNIST(
        root = "4_computer_vision/datasets/",
        train = False,
        download = True,
        transform = transforms.ToTensor()
    )
    _, w, h = train_ds.data.shape
    classes = train_ds.classes

    train_dl = DataLoader(train_ds, 
                          batch_size = BATCH_SIZE, 
                          shuffle = True,
                          num_workers = 8)
    test_dl = DataLoader(test_ds, 
                         batch_size = BATCH_SIZE, 
                         shuffle = True,
                         num_workers = 8)

    models = []
    models.append(FashionMNISTModelV1(input_size = w * h,
                                      hidden_size = 10,
                                      output_size = len(classes)))
    models.append(FashionMNISTModelV2(input_size = w * h,
                                      hidden_size = 10,
                                      output_size = len(classes)))
    loss_fn = torch.nn.CrossEntropyLoss()

    for i, model in enumerate(models):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), LR)
        tqdm.write(f"\nTraining MODEL {i + 1}:\n")
        train_time_start = time.time()
        for epoch in tqdm(range(EPOCHS)):
            train_step(model, optimizer, loss_fn, train_dl, device, train_losses, train_accuracies)
            test_step(model, loss_fn, test_dl, device, test_losses, test_accuracies)
            tqdm.write(f"Epoch: {epoch + 1}/{EPOCHS}\n------------\n" \
                       f"Train loss: {train_losses[-1] :.4f}\t|\t" \
                       f"Train accuracy: {(train_accuracies[-1]) * 100 :2.2f}% \n" \
                       f"Test loss: {test_losses[-1] :.4f}\t|\t" \
                       f"Test accuracy: {(test_accuracies[-1]) * 100 :2.2f}%\n")
        train_time = int(time.time() - train_time_start)
        print(f"Training time: {datetime.timedelta(seconds = train_time)}s\n")