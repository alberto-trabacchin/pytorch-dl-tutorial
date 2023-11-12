import torch

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
    weight = 0.2
    bias = 0.1
    X = torch.arange(start = 1, end = 10, step = 0.01).unsqueeze(dim = 1)
    y = weight * X + bias
    model = LRModel()
    predictions = model(X)
    print(predictions)
                                          