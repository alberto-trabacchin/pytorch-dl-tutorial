import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchinfo

if __name__ == "__main__":
    model = resnet50(
        weights = ResNet50_Weights.DEFAULT,
        progress = True
    )
    print(model)