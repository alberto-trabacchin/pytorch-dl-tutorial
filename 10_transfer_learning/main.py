import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchinfo

if __name__ == "__main__":
    model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 3)
    )
    torchinfo.summary(
        model = model,
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )