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
        self.linear_layer = torch.nn.Linear(in_features = 1, out_features = 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    weight = 0.2
    bias = 0.1
    X = torch.arange(start = 1, end = 10, step = 0.1, dtype = torch.float32).unsqueeze(dim = 1)
    y = weight * X + bias
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    model = LRModel()
    model.to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    epochs = int(2e3)
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
                train_losses.append(train_loss.cpu().detach().numpy())
                test_losses.append(test_loss.cpu().detach().numpy())
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
    MODEL_NAME = "lr_model_v2.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plot_losses(train_losses, test_losses, epoch_count)
    plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), y_pred.cpu())
    print(model.state_dict())
    plt.show()

    loaded_model = LRModel()
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    loaded_model.eval()
    with torch.inference_mode():
        y_loaded_pred = loaded_model(X_test)
    plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), y_pred.cpu())
    print(f"Correct loaded-model predictions: {(y_pred == y_loaded_pred).all()}")
    plt.show()