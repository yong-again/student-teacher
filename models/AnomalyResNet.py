"""
AnomalyResNet.py
Using pretrained model(resnet) for Teacher Networks
"""



import torch
from torchvision.models import ResNet18_Weights
from torchvision import models
import torch.nn as nn


class AnomalyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = self._get_backbone()
        self.Linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def _get_backbone(self):
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # load pretrain model
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1]) # remove classifier
        return resnet18

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model = AnomalyResNet()
    dummy = torch.randn(16, 3, 65, 65)

    preds = model(dummy)
    print(preds.shape)