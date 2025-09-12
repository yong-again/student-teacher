from numpy.f2py.capi_maps import getarrdims
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import warnings
warnings.filterwarnings("ignore")

class AnomalyResNet(nn.Module):
    def __init__(self, model_name: str = 'resnet18'):
        super(AnomalyResNet, self).__init__()
        self.model_name = model_name
        self.model = self._get_backbone()
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def _get_backbone(self):
        model_class = getattr(models, self.model_name)
        model = model_class(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])

        return model

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        y = self.softmax(self.linear(x))

        return y

if __name__ == "__main__":
    from torchsummary import summary
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    resnet34 = AnomalyResNet(model_name='resnet34')
    print("load model : ", resnet34.model_name)
    print("model structure : ",)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet34.to(device)
    summary(resnet34, (3, 65, 65))

    resnet34 = nn.Sequential(*list(resnet34.children())[:-2])
    print("model structure after remove last 2 layers : ",)

    summary(resnet34, (3, 65, 65))


