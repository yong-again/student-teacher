import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn



class AnomalyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self._get_resnet_backbone()
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def _get_resnet_backbone(self):
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet18 = nn.Sequential(*list(resnet18.children()))[:-1]
        return resnet18

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = AnomalyResNet()
    dummy = torch.randn(1, 3, 65, 65)

    preds = model(dummy)
    print(preds.size())



