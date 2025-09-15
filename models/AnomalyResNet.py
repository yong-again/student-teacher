from numpy.f2py.capi_maps import getarrdims
from torch import nn
from torchvision import models
try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
except Exception as e:
    ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = ResNet101_Weights = None
import warnings
warnings.filterwarnings("ignore")

class AnomalyResNet(nn.Module):
    def __init__(self, model_name: str = 'resnet18',  num_classes: int=None):
        super().__init__()
        self.model_name = model_name

        # 1) make backbone = ImageNet Weights
        model_ctor = getattr(models, model_name)
        if "18" in model_name:
            weights = ResNet18_Weights.IMAGENET1K_V1 if ResNet18_Weights else None
        elif "34" in model_name:
            weights = ResNet34_Weights.IMAGENET1K_V1 if ResNet34_Weights else None
        elif "50" in model_name:
            weights = ResNet50_Weights.IMAGENET1K_V1 if ResNet50_Weights else None
        elif "101" in model_name:
            weights = ResNet101_Weights.IMAGENET1K_V1 if ResNet101_Weights else None
        else:
            weights = None
        backbone = model_ctor(weights=weights)

        # 2) modify the last layer -> for feature extraction, remove fc layer
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

        # 3) optional
        self.head = nn.Linear(feat_dim, num_classes) if num_classes is not None else None

    def forward(self, x, return_features: bool = True):
        feats = self.backbone(x).flatten(1) # [B, feat_dim]
        if return_features or self.head is None:
            return feats

        logits = self.head(feats)
        return logits

if __name__ == "__main__":
    from torchsummary import summary
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    resnet18 = AnomalyResNet(model_name='resnet18')
    print("load model : ", resnet18.model_name)
    print("model structure : ",)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18.to(device)
    summary(resnet18, (3, 65, 65))


