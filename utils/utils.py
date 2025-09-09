import torch
import os
import traceback
import torch.nn as nn
from datetime import datetime
from pathlib import Path

def load_model(model, path):
    try:
        model_name = os.path.basename(path)
        model.load_state_dict(torch.load(path))

    except FileNotFoundError as e:
        print(traceback.format_exc())
        print(f"Failed to load {path}")

def get_model_path(root_dir, category, type, patch_size=None, number=None):
    model_weights_base_path = os.path.join(root_dir, 'weights', category)
    if type.lower() == 'resent18':
        return os.path.join(model_weights_base_path, f'resnet18_{category}.pth')
    elif type == 'teacher' and patch_size is not None:
        return os.path.join(model_weights_base_path, f'teacher_{patch_size}.pth')
    elif type == 'student' and patch_size is not None:
        return os.path.join(model_weights_base_path, f'student_{patch_size}_{number}.pth')

def increment_mean_and_var(mu_N, var_N, N, outputs):
    # batch: [batch, h, w, vector]
    B = outputs.size()[0]
    # descriptor vector -> mean over batch and pixels
    mu_B = torch.mean(outputs, dim=[0, 1, 2])
    S_B = B * torch.var(outputs, dim=[0, 1, 2], unbiased=False)
    S_N = N * var_N
    mu_NB = N / (N + B) * mu_N + B / (N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N + B)
    return mu_NB, var_NB, N + B

def mc_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout)\
                or isinstance(module, nn.Dropout2d)\
                or isinstance(module, nn.Dropout3d):
            module.train()

def create_experiment_path(root_path, root="results", category="basic"):
    """
    모델 훈련 결과 및 테스트 결과 저장 경로 생성
    """
    date_dir = datetime.now().strftime("%Y%m%d")
    base_dir = root_path / Path(root) / date_dir / category
    base_dir.mkdir(parents=True, exist_ok=True)

    exp_num = 1
    while True:
        exp_dir = base_dir / (f"exp" if exp_num == 1 else f"exp{exp_num}")
        if not exp_dir.exists():
            exp_dir.mkdir()
            return exp_dir
        exp_num += 1
