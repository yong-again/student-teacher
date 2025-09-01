from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class AnomalyResNet(nn.Module):
    def __init__(self,):
        super(AnomalyResNet, self).__init__()
        self.resnet18 = self._get_backbone()
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)


    def _get_backbone(self):
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        return resnet18

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(-1, 512)
        y = self.softmax(self.linear(x))

        return y


if __name__ == "__main__":
    from torchsummary import summary
    import torch
    resnet18 = AnomalyResNet()

    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    summary(resnet18, (3, 65, 65))

