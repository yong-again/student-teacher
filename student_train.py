import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from einops import rearrange, repeat
from torchvision import transforms

from dataset.AnomalyDataset import AnomalyDataset
from models.AnomalyNet import AnomalyNet
from models.AnomalyResNet import AnomalyResNet
from utils.utils import load_model, get_model_path, increment_mean_and_var, mc_dropout
from utils.loss import knowledge_distillation, compactness_loss, student_loss
from configs.config import TrainStudent


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    CONFIG = TrainStudent()

    # define teacher network
    teacher = AnomalyNet().create((CONFIG.patch_size, CONFIG.patch_size))
    teacher.to(device).eval()

    # load teacher model
    model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'teacher', CONFIG.patch_size)
    print(model_path)
    if os.path.exists(model_path):
        load_model(teacher, model_path)
        print(f"Successfully loaded: {model_path}")

    # students networks
    students = [AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size)) for _ in range(CONFIG.n_students)]
    students = [student.to(device) for student in students]

    # define optimizer
    optimizer = [optim.Adam(student.parameters(),
                            lr=CONFIG.learning_rate,
                            weight_decay=CONFIG.weight_decay) for student in students]

    # Load dataset
    dataset = AnomalyDataset(
        root_dir=CONFIG.root_dir,
        transforms=transforms.Compose([
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.RandomCrop((CONFIG.patch_size, CONFIG.patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
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

    # compute incremental mean an var over training set
    # because the whole training set takes too much memory space
    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, (images, labels) in enumerate(tqdm(dataloader)):
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



if __name__ == "__main__":
    train()

