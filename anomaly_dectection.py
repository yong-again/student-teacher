import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, reduce
import os
import torch.nn.functional as F

from models.AnomalyNet import AnomalyNet
from dataset.AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model, get_model_path, create_experiment_path
from sklearn.metrics import roc_curve, auc
from configs import Inference

# ---------------
# ERR & VAR MAPS
# ---------------
def get_err_map(students_pred, teacher_pred):
    # students_pred: [b, id , h, w, vec]
    # teacher_pred:  [b,    ,h,  w, vec]
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    err = reduce((mu_students - teacher_pred) ** 2, 'b h w vec -> b h w', 'sum')
    return err

def get_variance_map(students_pred):
    # students_pred: [b, id , h, w, vec]
    # Var over students (id) of the feature vector, summed over vec per pixel
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    # E_id[X^2] (id에 대한 평균) -> [b, h, w, vec]
    ex2_idmean = reduce(students_pred ** 2, 'b id h w vec -> b h w vec', 'mean')
    # Var_id per pixel (sum over vec): E[X^2] - (E[X])^2
    ex2 = reduce(ex2_idmean, 'b h w vec -> b h w', 'sum')
    emu2 = reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
    var = ex2 - emu2
    return var

# -----------------
# CALIBRATION
# -----------------
@torch.no_grad()
def calibrate(teacher, students, dataloader, device, use_l2_norm: bool = True):
    print('Calibrating teacher on students')

    # 1) Teacher feature 통계
    t_mu, t_var, t_N = 0, 0, 0
    for _, (images, _, _) in enumerate(tqdm(dataloader)):
        inputs = images.to(device, non_blocking=True)
        t_out = teacher.fdfe(inputs)                     # [..., vec]
        if use_l2_norm:
            t_out = F.normalize(t_out, dim=-1)
        t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)

    # 2) 에러/분산 통계 및 최대값
    mu_err, var_err, N_err = 0, 0, 0
    mu_var, var_var, N_var = 0, 0, 0
    max_err = torch.tensor(0.0, device=device)
    max_var = torch.tensor(0.0, device=device)

    for images, _, _ in tqdm(dataloader):
        inputs = images.to(device, non_blocking=True)

        t_out = teacher.fdfe(inputs)
        if use_l2_norm:
            t_out = F.normalize(t_out, dim=-1)
        # 표준화 eps는 1e-6 (기존 코드의 1e+6 오타 수정)
        t_out = (t_out - t_mu) / torch.sqrt(t_var + 1e-6)

        s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)  # [b,id,h,w,vec] 또는 [b,id,vec]
        if use_l2_norm:
            s_out = F.normalize(s_out, dim=-1)

        s_err = get_err_map(s_out, t_out)   # [b,h,w]
        s_var = get_variance_map(s_out)     # [b,h,w]

        mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
        mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

        max_err = torch.maximum(max_err, torch.max(s_err))
        max_var = torch.maximum(max_var, torch.max(s_var))

    # ← 루프 바깥에서 반환 (조기 return 버그 수정)
    return {
        "teacher": {"mu": t_mu, "var": t_var},
        "students": {
            "err": {"mu": mu_err, "var": var_err, "max": max_err},
            "var": {"mu": mu_var, "var": var_var, "max": max_var},
        },
    }

# ---------------------------
# SCORE MAP
# ---------------------------
@torch.no_grad()
def get_score_map(inputs, teacher, students, params, use_l2_norm: bool = True):
    t_out = teacher.fdfe(inputs)
    if use_l2_norm:
        t_out = F.normalize(t_out, dim=-1)
    t_out = (t_out - params['teacher']['mu']) / torch.sqrt(params['teacher']['var'] + 1e-6)

    s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)
    if use_l2_norm:
        s_out = F.normalize(s_out, dim=-1)

    s_err = get_err_map(s_out, t_out)
    s_var = get_variance_map(s_out)

    score_map = ((s_err - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'] + 1e-6)
               + (s_var - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'] + 1e-6))
    return score_map

# ---------------------------
# VISUALIZATION
# ---------------------------
def visualize(img, gt, score_map, max_score, save_path=None):
    plt.figure(figsize=(13, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
    plt.colorbar()
    plt.title("Anomaly Score Map")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# ---------------------------
# MAIN
# ---------------------------
def detect_anomaly():
    CONFIG = Inference()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Teacher
    teacher = AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size))
    teacher.eval().to(device)

    teacher_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'teacher', patch_size=CONFIG.patch_size)
    if os.path.exists(teacher_model_path):
        load_model(teacher, teacher_model_path)
        print(f"Successfully teacher model loaded: {teacher_model_path}")

    # Students
    students = [AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size)) for _ in range(CONFIG.n_students)]
    students = [student.eval().to(device) for student in students]
    for i in range(CONFIG.n_students):
        student_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'student',
                                            patch_size=CONFIG.patch_size, number=i)
        if os.path.exists(student_model_path):
            load_model(students[i], student_model_path)
            print(f"Successfully student model loaded: {student_model_path}")

    # -------- 전처리 일관성: ImageNet 통계로 통일 --------
    image_transforms = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    # 시각화용 unnormalize (ImageNet 기준)
    unorm = transforms.Normalize(mean=(-1, -1, -1),
                                 std=(2, 2, 2))

    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),
    ])

    # Calibration on anomaly-free train
    calib_dataset = AnomalyDataset(CONFIG.root_dir, category=CONFIG.category, split='train',
                                   transform=image_transforms, mask_transform=mask_transform)
    calib_dataloader = DataLoader(calib_dataset, batch_size=CONFIG.batch_size,
                                  shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True)
    # 학습과 동일 규약: L2 정규화 사용(True/False를 학습 설정과 매칭)
    use_l2_norm = True
    params = calibrate(teacher, students, calib_dataloader, device, use_l2_norm=use_l2_norm)

    # Test
    test_dataset = AnomalyDataset(root_dir=CONFIG.root_dir, transform=image_transforms,
                                  mask_transform=mask_transform, category=CONFIG.category, split='test')
    test_dataloader= DataLoader(test_dataset, shuffle=False, batch_size=CONFIG.batch_size,
                                num_workers=CONFIG.num_workers, pin_memory=True)

    y_score, y_true = np.array([]), np.array([])
    result_dir = create_experiment_path(CONFIG.root_dir, category=CONFIG.category)
    print(f"\nSaving results to {result_dir}")
    os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)

    img_count = 0
    for images, labels, masks in tqdm(test_dataloader):
        inputs = images.to(device, non_blocking=True)
        gt_masks = masks.to(device, non_blocking=True)

        # get_score_map에서도 동일하게 use_l2_norm 적용
        score_map = get_score_map(inputs, teacher, students, params, use_l2_norm=use_l2_norm)
        y_score = np.concatenate([y_score, rearrange(score_map, 'b h w -> (b h w)').cpu().numpy()])
        y_true = np.concatenate([y_true, rearrange(gt_masks, 'b c h w -> (b h w c)').cpu().numpy()])

        if CONFIG.visualize:
            img_in = rearrange(unorm(inputs), 'b c h w -> b h w c').cpu()
            gt_in  = rearrange(gt_masks, 'b c h w -> b h w c').cpu()

            # max_score도 eps 포함해 일관 계산
            max_err = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'] + 1e-6)
            max_var = (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'] + 1e-6)
            max_score = (max_err + max_var).item()

            for b in range(inputs.shape[0]):
                save_path = os.path.join(result_dir, f"images/img_{b:02d}{img_count}.png")
                visualize(img_in[b].squeeze(),
                          gt_in[b].squeeze(),
                          score_map[b].squeeze().cpu().numpy(),
                          max_score,
                          save_path=save_path)
                img_count += 1

    # Binarize GT and compute AUROC
    y_true = np.where(y_true > 0.5, 1, 0).astype(int)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        pixel_auroc = auc(fpr, tpr)
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC (area={pixel_auroc:.4f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {CONFIG.category}')
        plt.legend(loc="lower right")
        plot_path = os.path.join(result_dir, 'roc_curve.png')
        plt.savefig(plot_path); plt.close()
        print(f"ROC curve plot saved to {plot_path}")
    else:
        print("Could not calculate AUROC. All ground truth labels are the same.")

    np.save(os.path.join(result_dir, 'y_score.npy'), y_score)
    np.save(os.path.join(result_dir, 'y_true.npy'), y_true)
    print("Results saved successfully.")

if __name__ == "__main__":
    detect_anomaly()