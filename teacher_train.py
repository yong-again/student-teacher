"""
teacher_train.py
Training for teacher network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
from einops import rearrange
from torchvision import transforms
import numpy as np

from dataset.AnomalyDataset import AnomalyDataset
from models.AnomalyNet import AnomalyNet
from models.AnomalyResNet import AnomalyResNet
from utils.utils import load_model, get_model_path
from utils.loss import knowledge_distillation, compactness_loss
from configs.config import TrainTeacher

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    CONFIG = TrainTeacher()

    # load resnet model for knowledge distillation
    resnet = AnomalyResNet(model_name=CONFIG.model_name)
    #load_model(resnet, model_path)
    # resnet = nn.Sequential(*list(resnet.children())[:-1])
    # print(resnet)
    resnet.to(device).eval().requires_grad_(False)
    #print(f"Successfully Model loaded: {model_path}")

    # Set Teacher Network
    teacher = AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size))
    teacher.to(device)

    # define optimizer
    optimizer = optim.Adam(teacher.parameters(),
                           lr=CONFIG.learning_rate,
                           weight_decay=CONFIG.weight_decay)

    dataset = AnomalyDataset(
        root_dir=CONFIG.root_dir,
        transform=transforms.Compose([
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.RandomCrop((CONFIG.patch_size, CONFIG.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]),
        split='train',
        category=CONFIG.category,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True
    )

    #train loop
    min_running_loss = np.inf
    lambda_k = 1.0
    lambda_c = 1.0
    for epoch in range(CONFIG.num_epochs):
        running_loss = 0.0

        for i, (images, labels, masks) in enumerate(tqdm(dataloader, desc="Training Teacher Network")):
            optimizer.zero_grad(set_to_none=True)

            inputs = images.to(device)
            with torch.no_grad():
                targets = resnet(inputs).detach() # h=w=1
            outputs = teacher(inputs)
            if outputs.dim() == 4:
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)

            targets_n = F.normalize(targets, dim=1)
            outputs_n = F.normalize(outputs, dim=1)

            Lk = knowledge_distillation(outputs_n, targets_n)
            Lc = compactness_loss(outputs)
            loss = lambda_k * Lk + lambda_c * Lc # Lt = Lk + Lm + Lc(1:0:1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{CONFIG.num_epochs} Loss: {running_loss / len(dataloader)}")

        if running_loss < min_running_loss and epoch > 0:
            model_path = get_model_path(CONFIG.root_dir, CONFIG.category, type='teacher', patch_size=CONFIG.patch_size)
            torch.save(teacher.state_dict(), model_path)
            print(f"Loss decreased: {min_running_loss / len(dataloader)} -> {running_loss / len(dataloader)}")
            print(f"Model saved at {model_path}")

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0

if __name__ == '__main__':
    train()






