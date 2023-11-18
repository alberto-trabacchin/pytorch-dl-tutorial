from prettytable import PrettyTable
import torch
from going_modular import model_builder
from typing import Dict, List


def accuracy(y_pred_logit, y_true_classes):
    y_pred_class = torch.softmax(y_pred_logit, dim = 1).argmax(dim = 1)
    return (y_pred_class == y_true_classes).sum().item() / len(y_true_classes)


def make_table_train_results(
        model_name: str,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float
) -> None:
    table = PrettyTable(["Model Name", "Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])
    table.add_row([model_name,
                   epoch,
                   f"{train_loss :.4f}",
                   f"{100 * train_acc :.2f}%",
                   f"{test_loss :.4f}",
                   f"{100 * test_acc :.2f}%"])
    return table


def print_models_results(models_results: Dict[str, dict]) -> None:
    table = PrettyTable(["Model Name", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])
    for model_name, model_result in models_results.items():
        table.add_row([model_name,
                       f"{model_result['train_loss'][-1] :.4f}",
                       f"{100 * model_result['train_acc'][-1] :.2f}%",
                       f"{model_result['test_loss'][-1] :.4f}",
                       f"{100 * model_result['test_acc'][-1] :.2f}%"])
    print(table)
    

def count_parameters(model: torch.nn.Module, verbose = True) -> int:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if verbose:
        print(table)
        print(f"Total trainable params: {total_params}")
    return total_params


if __name__ == "__main__":
    model = model_builder.TinyVGG(
        input_shape = 3,
        hidden_units = 32,
        output_shape = 3
    )
    count_parameters(model)