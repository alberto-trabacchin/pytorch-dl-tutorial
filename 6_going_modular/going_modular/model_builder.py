import torch
from going_modular import utils

class TinyVGG(torch.nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, name: str) -> None:
        super(TinyVGG, self).__init__()
        self.name = name
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = input_shape, 
                            out_channels = hidden_units, 
                            kernel_size = 3,
                            stride = 1, 
                            padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_units, 
                            out_channels = hidden_units, 
                            kernel_size = 3,
                            stride = 1,
                            padding = 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        )
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = hidden_units, 
                            out_channels = hidden_units, 
                            kernel_size = 3,
                            stride = 1,
                            padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = hidden_units, 
                            out_channels = hidden_units, 
                            kernel_size = 3,
                            stride = 1,
                            padding = 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 1690, 
                            out_features = output_shape),
            torch.nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVGG(
        input_shape = 3,
        hidden_units = 32,
        output_shape = 3
    ).to(device)
    utils.count_parameters(model)