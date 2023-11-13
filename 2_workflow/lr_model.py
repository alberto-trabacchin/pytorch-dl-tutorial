import torch
import matplotlib.pyplot as plt
from pathlib import Path

def plot_predictions(X_train, y_train, X_test, y_test, y_pred = None):
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, label = "Train", color = "blue", s = 5)
    ax.scatter(X_test, y_test, label = "Test", color = "green", s = 5)
    if y_pred is not None:
        ax.scatter(X_test, y_pred, label = "Predictions", color = "red", s = 5)
    fig.legend(loc = "upper right", fontsize = 12)


def plot_losses(train_losses, test_losses, epoch_count):
    fig, ax = plt.subplots()
    ax.plot(epoch_count, train_losses, label = "Training")
    ax.plot(epoch_count, test_losses, label = "Testing")
    fig.legend(loc = "upper right", fontsize = 12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")


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
    X = torch.arange(start = 1, end = 10, step = 0.1, dtype = torch.float32).unsqueeze(dim = 1)
    y = weight * X + bias
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    model = LRModel()
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    epochs = 100
    train_losses = []
    test_losses = []
    epoch_count = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        train_loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            if (epoch % 10) == 0:
                y_pred = model(X_test)
                test_loss = loss_fn(y_pred, y_test)
                train_losses.append(train_loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())
                epoch_count.append(epoch)
                print(f"Epoch {epoch}")
                print("------------------")
                print(f"Train loss: {train_loss :.4f}")
                print(f"Test loss: {test_loss :.4f} \n")
    
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_test)

    MODEL_PATH = Path("2_workflow/models")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = "model_1_lr.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_losses(train_losses, test_losses, epoch_count)
    plot_predictions(X_train, y_train, X_test, y_test, y_pred)
    print(model.state_dict())
    plt.show()