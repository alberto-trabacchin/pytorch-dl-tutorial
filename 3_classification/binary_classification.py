import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("3_classification/helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("3_classification/helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


class BinaryClassification(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassification, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias = True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1, bias = True)
        )

    def forward(self, x):
        return self.layers(x)
    

def accuracy(y_pred, y_true):
    y_pred = torch.round(torch.sigmoid(y_pred))
    return (y_pred == y_true).sum().item() / len(y_true)


def plot_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Spectral, s = 18)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = make_circles(n_samples = int(1e3), noise = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    model = BinaryClassification(input_size = 2, hidden_size = 10)
    model.to(device)
    X_train_ts = torch.from_numpy(X_train).float().to(device)
    y_train_ts = torch.from_numpy(y_train).float().to(device)
    X_test_ts = torch.from_numpy(X_test).float().to(device)
    y_test_ts = torch.from_numpy(y_test).float().to(device)
    loss_fun = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    epochs = int(5e4)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_counter = []

    for epoch in range(epochs):
        model.train()
        y_pred_train = model(X_train_ts).squeeze()
        optimizer.zero_grad()
        train_loss = loss_fun(y_pred_train, y_train_ts)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            if epoch % 100 == 0:
                y_pred_test_tensor = model(X_test_ts).squeeze()
                test_loss = loss_fun(y_pred_test_tensor, y_test_ts)
                test_accuracy = accuracy(y_pred_test_tensor, y_test_ts)
                train_accuracy  = accuracy(y_pred_train, y_train_ts)
                train_losses.append(train_loss.cpu().numpy())
                test_losses.append(test_loss.cpu().numpy())
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                epoch_counter.append(epoch)
                print(f"Epoch: {epoch}/{epochs}")
                print("------------------------------------------------")
                print(f"Train loss: {train_loss :.4f} \t Test loss: {test_loss :.4f}")
                print(f"Train accuracy: {train_accuracy :.4f} \t Test accuracy: {test_accuracy:.4f}\n")

    MODEL_PATH = Path("3_classification/models")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = "binary_class_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    loaded_model = BinaryClassification(input_size = 2, hidden_size = 10)
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    loaded_model.eval()
    with torch.inference_mode():
        y_loaded_pred_tensor = loaded_model(X_test_ts).squeeze()
    print(f"Correct loaded-model predictions: {(y_pred_test_tensor == y_loaded_pred_tensor).all()}")

    plot_dataset(X, y)
    plot_accuracies(train_accuracies, test_accuracies, epoch_counter)
    plot_losses(train_losses, test_losses, epoch_counter)
    plot_dec_bounds(X_train_ts, X_test_ts, y_train_ts, y_test_ts, model)
    plt.show()