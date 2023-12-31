import torch
import matplotlib.pyplot as plt

def plot_predictions(X_train, y_train, X_test, y_test, y_pred = None):
    plt.scatter(X_train, y_train, label = "Train", color = "blue", s = 4)
    plt.scatter(X_test, y_test, label = "Test", color = "green", s = 4)
    if y_pred is not None:
        plt.scatter(X_test, y_pred, label = "Predictions", color = "red", s = 4)
    plt.legend(prop = {"size": 12})
    plt.show()

if __name__ == "__main__":
    X = torch.arange(start = 1, end = 10, step = 0.1).unsqueeze(dim = 1)
    weight = 0.2
    bias = 2
    y = weight * X + bias
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    plot_predictions(X_train, y_train, X_test, y_test)