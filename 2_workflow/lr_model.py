import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(X_train, y_train, X_test, y_test, y_pred = None):
    plt.scatter(X_train, y_train, label = "Train", color = "blue", s = 5)
    plt.scatter(X_test, y_test, label = "Test", color = "green", s = 5)
    if y_pred is not None:
        plt.scatter(X_test, y_pred, label = "Predictions", color = "red", s = 5)
    plt.legend(prop = {"size": 12})
    plt.show()

class LRModel(torch.nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.weights = torch.nn.Parameter( torch.randn(1,
                                           requires_grad=True,
                                           dtype = torch.float ))
        self.bias = torch.nn.Parameter( torch.randn(1,
                                        requires_grad=True ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


if __name__ == "__main__":
    torch.manual_seed(42)
    weight = 0.2
    bias = 0.1
    X = torch.arange(start = 1, end = 10, step = 0.1).unsqueeze(dim = 1)
    y = weight * X + bias
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    model = LRModel()
    with torch.inference_mode():
        y_predict = model(X_test)
    print(list(model.parameters()))
    print(model.state_dict())
    plot_predictions(X_train, y_train, X_test, y_test, y_predict)
    