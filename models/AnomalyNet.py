import torch
import torch.nn as nn
from .FDFE import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool

class AnomalyNet:
    def __init__(self):
        self.patch_cnn = {
            **dict.fromkeys([(65, 65), 65, 'big'], AnomalyNet65),
            **dict.fromkeys([(33, 33), 33, 'medium'], AnomalyNet33),
            **dict.fromkeys([(17, 17), 17, 'small'], AnomalyNet17),
        }

    @classmethod
    def create(cls, model_name):
        self = cls()
        model = self.patch_cnn[model_name]
        if not model:
            raise ValueError('Invalid model name')

        return model()

class AnomalyNet65(nn.Module):
    patch_size = 65
    def __init__(self):
        super(AnomalyNet65, self).__init__()
        self.patch_width = 65
        self.patch_height = 65
        self.multiPoolPrepare = multiPoolPrepare(self.patch_height, self.patch_width)

        self.conv1 = nn.Conv2d(3, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 256, 4, 1)
        self.conv5 = nn.Conv2d(256, 128, 1, 1)
        self.output_channels = self.conv5.out_channels
        self.decode = nn.Linear(128, 512)

        self.dropout_2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.multiMaxPooling  = multiMaxPooling(2, 2,2,2 )
        self.unwrapPrepare = unwrapPrepare()

        self.l_relu = nn.LeakyReLU(5e-3)

    def fdfe(self, x):
        imH = x.size(2)
        imW = x.size(3)

        unwrapPool3 = unwrapPool(self.output_channels, imH / (2 * 2 * 2), imW / (2 * 2 * 2), 2, 2)
        unwrapPool2 = unwrapPool(self.output_channels, imH / (2 * 2), imW / (2 * 2), 2, 2)
        unwrapPool1 = unwrapPool(self.output_channels, imH / 2, imW / 2, 2, 2)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv3(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))

        x = self.unwrapPrepare(x)
        x = unwrapPool3(x)
        x = unwrapPool2(x)
        x = unwrapPool1(x)

        y = x.view(self.output_channels, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))

        return y

    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.patch_height and x.size(3) == self.patch_width, \
                    f"This patch extractor must be input of size (batch, 3, {self.patch_width}, {self.patch_height})"

            x = self.l_relu(self.conv1(x))
            x = self.max_pool(x)

            x = self.l_relu(self.conv2(x))
            x = self.max_pool(x)

            x = self.l_relu(self.conv3(x))
            x = self.max_pool(x)

            x = self.l_relu(self.conv4(x))
            x = self.l_relu(self.conv5(x))
            x = self.dropout(x)

            x = x.view(-1, self.output_channels)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)

            return x

class AnomalyNet33(nn.Module):
    patch_size = 33
    def __init__(self):
        super(AnomalyNet33, self).__init__()
        self.patch_width = 33
        self.patch_height = 33
        self.multiPoolPrepare = multiPoolPrepare(self.patch_height, self.patch_width)

        self.conv1 = nn.Conv2d(3, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 256, 5, 1)
        self.conv3 = nn.Conv2d(256, 256, 2, 1)
        self.conv4 = nn.Conv2d(256, 128, 4, 1)
        self.output_channels = self.conv4.out_channels
        self.decode = nn.Linear(128, 512)

        self.dropout_2d = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.multiMaxPooling = multiMaxPooling(2, 2,2, 2)
        self.unwrapPrepare = unwrapPrepare()

        self.l_relu = nn.LeakyReLU(5e-3)

    def fdfe(self, x):
        imH = x.size(2)
        imW = x.size(3)

        unwrapPool2 = unwrapPool(self.output_channels, imH / (2 * 2), imW / (2 * 2), 2, 2)
        unwrapPool1 = unwrapPool(self.output_channels, imH / 2, imW / 2, 2, 2)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv3(x))
        x = self.l_relu(self.conv4(x))

        x = self.unwrapPrepare(x)
        x = unwrapPool2(x)
        x = unwrapPool1(x)

        y = x.view(self.output_channels, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))
        return y


    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.patch_height and x.size(3) == self.patch_width, \
                    f"This patch extractor must be input of size (batch, 3, {self.patch_width}, {self.patch_height})"

            x = self.l_relu(self.conv1(x))
            x = self.max_pool(x)

            x = self.l_relu(self.conv2(x))
            x = self.max_pool(x)

            x = self.l_relu(self.conv3(x))
            x = self.l_relu(self.conv4(x))

            x = self.dropout_2d(x)
            x = x.view(-1, self.output_channels)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)

            return x

class AnomalyNet17(nn.Module):
    patch_size = 17
    def __init__(self):
        super(AnomalyNet17, self).__init__()
        self.patch_width = 17
        self.patch_height = 17
        self.multiPoolPrepare = multiPoolPrepare(self.patch_width, self.patch_height)

        self.conv1 = nn.Conv2d(3, 128, 6, 1)
        self.conv2 = nn.Conv2d(128, 256, 5, 1)
        self.conv3 = nn.Conv2d(256, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 128, 4, 1)
        self.output_channels = self.conv4.out_channels
        self.decode = nn.Linear(128, 512)
        self.dropout_2d = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.l_relu = nn.LeakyReLU(5e-3)


    def fdfe(self, x):
        imH = x.size(2)
        imW = x.size(3)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.l_relu(self.conv2(x))
        x = self.l_relu(self.conv3(x))
        x = self.l_relu(self.conv4(x))

        y = x.view(self.output_channels, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))

        return y

    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.patch_height and x.size(3) == self.patch_width, \
                    f"This patch extractor must be input of size (batch, 3, {self.patch_width}, {self.patch_height})"
            x =  self.l_relu(self.conv1(x))
            x = self.l_relu(self.conv2(x))
            x = self.l_relu(self.conv3(x))
            x = self.l_relu(self.conv4(x))
            x = self.dropout_2d(x)
            x = x.view(-1, self.output_channels)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)
            return x

if __name__ == '__main__':
    from torchvision import models
    from torchvision.models import ResNet18_Weights

    pH = 65
    pW = 65

    resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

    x = torch.randn(6, 3, pH, pW)

    teacher = AnomalyNet().create((pH, pW))
    teacher_fdfe = teacher.fdfe(x)


    y_net = teacher_fdfe(x)
    y_resnet18 = resnet18(x)

    print(y_net.size())
    print(torch.squeeze(y_resnet18).size())



















