import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import requests
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


# Download helper functions from Learn PyTorch repo (if not already downloaded)
Path.mkdir(Path("utils"), exist_ok = True)
if Path("utils/helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("utils/helper_functions.py", "wb") as f:
        f.write(request.content)

from utils.helper_functions import accuracy_fn


class TinyVGGModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int = 1):
        super().__init__()
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = input_size,
                            out_channels = hidden_size,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_size,
                            out_channels = hidden_size,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2))
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = hidden_size,
                            out_channels = hidden_size,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_size,
                            out_channels = hidden_size,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 735,
                            out_features = output_size)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


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

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: str = "cpu"):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for X in data:
            X = torch.unsqueeze(X, dim = 0).to(torch.float32).to(device)
            y_pred_logit = model(X).squeeze(dim = 0)
            y_pred_prob = torch.softmax(y_pred_logit, dim = 0)
            predictions.append(y_pred_prob.to("cpu"))
    return torch.stack(predictions)
        

if __name__ == "__main__":
    EPOCHS = 0
    LR = 0.1
    BATCH_SIZE = 256
    HIDDEN_SIZE = 15
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path.mkdir(Path("datasets/"), exist_ok = True)

    # Download FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(
        root = "datasets/",
        train = True,
        download = True,
        transform = transforms.ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
        root = "datasets/",
        train = False,
        download = True,
        transform = transforms.ToTensor()
    )

    # Create dataloaders
    train_dl = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    # Create model
    model = TinyVGGModel(
        input_size = 1,
        hidden_size = HIDDEN_SIZE,
        output_size = len(train_dataset.classes)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(EPOCHS)):
        train_step(
            model,
            optimizer,
            loss_fn,
            train_dl,
            device,
            train_losses,
            train_accuracies
        )
        test_step(
            model,
            loss_fn,
            test_dl,
            device,
            test_losses,
            test_accuracies
        )
        tqdm.write(
            f"Epoch: {epoch + 1}/{EPOCHS}\n------------\n" \
            f"Train loss: {train_losses[-1] :.4f}\t|\t" \
            f"Train accuracy: {(train_accuracies[-1]) * 100 :2.2f}% \n" \
            f"Test loss: {test_losses[-1] :.4f}\t|\t" \
            f"Test accuracy: {(test_accuracies[-1]) * 100 :2.2f}%\n"
        )
    
    # Make predictions (dim X_pred != N x C x H x W = 10000 x 1 x 28 x 28)
    X_pred = list(torch.unsqueeze(test_dataset.data, dim = 1).float())
    pred_prob = make_predictions(model, X_pred, device)
    pred_class = pred_prob.argmax(dim = 1)
    
    # Calculate confusion matrix
    confmat = ConfusionMatrix(num_classes = len(test_dataset.classes), task = "multiclass")
    confmat_tensor = confmat(preds = pred_class, target = test_dataset.targets)
    fig, ax = plot_confusion_matrix(
        conf_mat = confmat_tensor.numpy(),
        class_names = test_dataset.classes,
        figsize = (10, 7)
    )
    plt.show()