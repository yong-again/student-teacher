import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from torchvision import transforms

from dataset.AnomalyDataset import AnomalyDataset
from models.AnomalyNet import AnomalyNet
from utils.utils import load_model, get_model_path, increment_mean_and_var
from utils.loss import student_loss
from configs.config import TrainStudent


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    CONFIG = TrainStudent()

    # define teacher network
    teacher = AnomalyNet().create((CONFIG.patch_size, CONFIG.patch_size))
    teacher.to(device).eval()

    # load teacher model
    teacher_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'teacher', CONFIG.patch_size)
    if os.path.exists(teacher_model_path):
        load_model(teacher, teacher_model_path)
        print(f"Successfully loaded: {teacher_model_path}")

    # students networks
    students = [AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size)) for _ in range(CONFIG.n_students)]
    students = [student.to(device) for student in students]

    # define optimizer
    optimizers = [optim.Adam(student.parameters(),
                            lr=CONFIG.learning_rate,
                            weight_decay=CONFIG.weight_decay) for student in students]

    # Load dataset
    dataset = AnomalyDataset(
        root_dir=CONFIG.root_dir,
        transform=transforms.Compose([
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        category=CONFIG.category
    )

    # Preprocessing
    # Apply teacher network on anomaly-free dataset
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
    )

    print(f"Preprocessing of training dataset {CONFIG.category}")

    # compute incremental mean a var over training set
    # because the whole training set takes too much memory space
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, (images, labels, masks) in enumerate(tqdm(dataloader)):
            inputs = images.to(device)
            t_out = teacher.fdfe(inputs)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)

    # training
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
    )

    for j, student in enumerate(students):
        min_running_loss = np.inf
        print(f'Training Student {j} on anomaly-free dataset ...')

        for epoch in range(CONFIG.num_epochs):
            running_loss = 0.0

            for i, (images, labels, masks) in enumerate(dataloader):
                optimizers[j].zero_grad()

                inputs = images.to(device)
                with torch.no_grad():
                    targets = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
                outputs = student.fdfe(inputs)
                loss = student_loss(targets, outputs)

                loss.backward()
                optimizers[j].step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print(f"Epoch [{epoch}/{CONFIG.num_epochs}], iter {i+1} \t Loss: {running_loss:.3f}")

                    if running_loss < min_running_loss and epoch > 0:
                        model_path = get_model_path(CONFIG.root_dir,
                                                    CONFIG.category,
                                                    'student',
                                                    CONFIG.patch_size,
                                                    j)
                        torch.save(student.state_dict(), model_path)
                        print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                        print(f"Model saved to {model_path}.")

                    min_running_loss = min(min_running_loss, running_loss)
                    running_loss = 0.0

if __name__ == "__main__":
    train()

