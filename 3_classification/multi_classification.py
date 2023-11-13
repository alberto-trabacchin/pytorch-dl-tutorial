import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from pathlib import Path 
import pandas as pd

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("3_classification/helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("3_classification/helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


class MultiClassification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiClassification, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias = True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias = True)
        )

    def forward(self, x):
        return self.layers(x)
    

def accuracy(y_pred, y_true):
    y_pred = torch.round(torch.sigmoid(y_pred))
    return (y_pred == y_true).sum().item() / len(y_true)


def plot_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu, s = 18)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_accuracies(train_accuracies, test_accuracies, epoch_counter):
    fig, ax = plt.subplots()
    ax.plot(epoch_counter, train_accuracies, label = "Training")
    ax.plot(epoch_counter, test_accuracies, label = "Testing")
    fig.legend(loc = "upper right", fontsize = 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")


def plot_losses(train_losses, test_losses, epoch_counter):
    fig, ax = plt.subplots()
    ax.plot(epoch_counter, train_losses, label = "Training")
    ax.plot(epoch_counter, test_losses, label = "Testing")
    fig.legend(loc = "upper right", fontsize = 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")


def plot_dec_bounds(X_train_ts, X_test_ts, y_train_ts, y_test_ts, model):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Spectral, s = 18)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision boundary")
    plot_decision_boundary(model, X = X_train_ts, y = y_train_ts)
    plot_decision_boundary(model, X = X_test_ts, y = y_test_ts)


if __name__ == "__main__":
    NUM_CLASSES = 5
    NUM_FEATURES = 2
    RAND_SEED = 42
    TRAIN_SIZE = 0.8
    LR = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RAND_SEED)
    X, y = make_blobs(n_samples = int(1e3), n_features = NUM_FEATURES, centers = NUM_CLASSES, cluster_std = 1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_SIZE, random_state = RAND_SEED)
    X_train_ts = torch.FloatTensor(X_train).to(device)
    X_test_ts = torch.FloatTensor(X_test).to(device)
    y_train_ts = torch.LongTensor(y_train).to(device)
    y_test_ts = torch.LongTensor(y_test).to(device)
    model = MultiClassification(input_size = NUM_FEATURES, hidden_size = 10, output_size = NUM_CLASSES)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LR)
    epochs = int(5e4)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_counter = []

    for epoch in range(epochs):
        model.train()
        y_pred_logit_train_ts = model(X_train_ts)
        y_pred_class_train_ts = torch.softmax(y_pred_logit_train_ts, dim = 1).argmax(dim = 1)
        optimizer.zero_grad()
        loss_train = loss_fn(y_pred_logit_train_ts, y_train_ts)
        loss_train.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.inference_mode():
                y_pred_logit_test_ts = model(X_test_ts)
                y_pred_class_test_ts = torch.softmax(y_pred_logit_test_ts, dim = 1).argmax(dim = 1)
                loss_test = loss_fn(y_pred_logit_test_ts, y_test_ts)
                train_losses.append(loss_train.item())
                test_losses.append(loss_test.item())
                epoch_counter.append(epoch)
                print(f"Epoch: {epoch}/{epochs}, train loss: {loss_train.item():.4f}, test loss: {loss_test.item():.4f}")
    
    plot_dec_bounds(X_train_ts, X_test_ts, y_train_ts, y_test_ts, model)
    plot_losses(train_losses, test_losses, epoch_counter)
    plt.show()
