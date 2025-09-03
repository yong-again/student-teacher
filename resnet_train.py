import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet18_Weights
import torch.optim as optim
import numpy as np
from configs.config import TrainResNet
from models.AnomalyResNet import AnomalyResNet
from dataset.AnomalyDataset import AnomalyResnetDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def train():
    config = TrainResNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sep = os.path.sep
    print(f"Device: {device}")

    resnet18 = AnomalyResNet()
    resnet18.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=config.learning_rate, momentum=config.momentum)

    resnet_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # load dataset
    dataset = AnomalyResnetDataset(
                    root_dir = config.root_dir,
                    transforms = resnet_transform,
                    category = config.category)

    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers)

    model_save_path = os.path.join(config.root_dir, 'weights', config.category)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_name = os.path.join(model_save_path, f'resnet18_{config.category}.pth')

    # train loop
    min_running_loss = np.inf
    for epoch in range(config.epochs):
        running_loss = 0.0
        running_corrects = 0
        max_running_corrects = 0

        for i, (img, labels) in enumerate(tqdm(dataloader, desc="Training ResNet for Knowledge Distillation")):
            optimizer.zero_grad()
            input = img.to(device)
            target = labels.to(device)
            outputs = resnet18(input)
            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            max_running_corrects += len(target)
            running_corrects += torch.sum(preds == target.data)

        print(f"{TrainResNet.category} Loss: {running_loss / len(dataloader):.4f}")
        accuracy = running_corrects.double() / max_running_corrects

        if running_loss < min_running_loss and epoch > 0:
            torch.save(resnet18.state_dict(), model_name)
            print(f"Loss decreased: {min_running_loss:.6f} -> {running_loss:.6f}.")
            print(f"Accuracy: {accuracy}")
            print(f"Model saved to {model_name}.")

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0

if __name__ == '__main__':
    train()

