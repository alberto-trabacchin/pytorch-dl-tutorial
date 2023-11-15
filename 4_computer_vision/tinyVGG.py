import torch


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
            torch.nn.Flatten(start_dim = 0, end_dim = -1),
            torch.nn.Linear(in_features = 640,
                            out_features = output_size)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_batch = torch.rand(32, 3, 32, 32, dtype = torch.float32)
    image = image_batch[0].to(device)
    model = TinyVGGModel(input_size = 3,
                         hidden_size = 10,
                         output_size = 10).to(device)
    output = model(image)
    print(output.shape)