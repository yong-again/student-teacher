import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange, reduce
import os


from models.AnomalyNet import AnomalyNet
from dataset.AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model, get_model_path, create_experiment_path
from sklearn.metrics import roc_curve, auc
from configs import Inference

def get_err_map(students_pred, teacher_pred):
    # student: [batch student_id, h, w, vector]
    # teacher: [batch, h ,w , vector]
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    err = reduce((mu_students - teacher_pred)**2, 'b h w vec -> b h w', 'sum')

    return err


def get_variance_map(students_pred):
    # student: [batch, student_id, h, w, vector]
    sse = reduce(students_pred, 'b id h w vec -> b id h w', 'mean')
    msse = reduce(sse, 'b id h w -> b h w', 'mean')
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    var = msse - reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
    return var

@torch.no_grad()
def calibrate(teacher, students, dataloader, device):
    print("Calibrating teacher on students")
    t_mu, t_var, t_N = 0, 0, 0
    for _, (images, labels, masks) in enumerate(tqdm(dataloader)):
        inputs = images.to(device)
        t_out = teacher.fdfe(inputs)
        t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)

    print("Calibrating scoring parameters on students")
    max_err, max_var = 0, 0
    mu_err, var_err, N_err = 0, 0, 0
    mu_var, var_var, N_var = 0, 0, 0

    for _, (images, labels, masks) in enumerate(tqdm(dataloader)):
        inputs = images.to(device)

        t_out = (teacher.fdfe(inputs) - t_mu) / torch.sqrt(t_var)
        s_out = torch.stack([student.fdfe(inputs) for student in students])

        s_err = get_err_map(s_out, t_out)
        s_var = get_variance_map(s_out)
        mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
        mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

        max_err = max(max_err, torch.max(s_err))
        max_var = max(max_var, torch.max(s_var))

        return {"teacher": {"mu": t_mu, "var": t_var},
                "students": {
                                "err": {"mu": mu_err, "var": var_err, "max": max_err},
                                "var": {"mu": mu_var, "var":var_var, "max": max_var}
                             }
                }

@torch.no_grad()
def get_score_map(inputs, teacher, students, params):
    t_out = (teacher.fdfe(inputs) - params['teacher']['mu']) / torch.sqrt(params['teacher']['var'])
    s_out = torch.stack([student.fdfe(inputs) for student in students], dim=1)

    s_err = get_err_map(s_out, t_out)
    s_var = get_variance_map(s_out)
    score_map = (s_err - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var']) \
                + (s_var - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var'])

    return score_map

def visualize(img, gt, score_map, max_score, save_path=None):
    """
    원본 이미지, Ground Truth, Score Map을 시각화하고 파일로 저장합니다.
    """
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
    
    # `save_path`가 제공되면 그림을 저장하고, 그렇지 않으면 화면에 표시합니다.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def detect_anomaly():
    CONFIG = Inference()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Teacher Network
    teacher = AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size))
    teacher.eval().to(device)

    # load teacher model
    teacher_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'teacher', CONFIG.patch_size)
    if os.path.exists(teacher_model_path):
        load_model(teacher, teacher_model_path)
        print(f"Successfully teacher model loaded: {teacher_model_path}")

    # students networks
    students = [AnomalyNet.create((CONFIG.patch_size, CONFIG.patch_size)) for _ in range(CONFIG.n_students)]
    students = [student.eval().to(device) for student in students]

    # loading students models
    for i in range(CONFIG.n_students):
        student_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'student', CONFIG.patch_size, i)
        if os.path.exists(student_model_path):
            load_model(students[i], student_model_path)
            print(f"Successfully student model loaded: {student_model_path}")

    image_transforms = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor()
    ])

    # calibration on anomaly-free dataset
    calib_dataset = AnomalyDataset(CONFIG.root_dir,
                                   category=CONFIG.category,
                                   split='train',
                                   transform=image_transforms,
                                   mask_transform=mask_transform
                                   )

    calib_dataloader = DataLoader(calib_dataset,
                                  batch_size=CONFIG.batch_size,
                                  shuffle=False,
                                  num_workers=CONFIG.num_workers)

    params = calibrate(teacher, students, calib_dataloader, device)

    # Load test dataset
    test_dataset = AnomalyDataset(root_dir=CONFIG.root_dir,
                                  transform=image_transforms,
                                  mask_transform=mask_transform,
                                  category=CONFIG.category,
                                  split='test'
                                  )
    
    test_dataloader= DataLoader(test_dataset, shuffle=False,
                                batch_size=CONFIG.batch_size,
                                num_workers=CONFIG.num_workers)


    y_score, y_true = np.array([]), np.array([])
    result_dir = create_experiment_path(CONFIG.root_dir, category=CONFIG.category)

    print(f"\nSaving results to {result_dir}")
    img_count = 0
    for images, labels, masks in tqdm(test_dataloader):
        inputs = images.to(device)
        gt_masks = masks.to(device)

        score_map = get_score_map(inputs, teacher, students, params)
        y_score = np.concatenate((y_score, rearrange(score_map, 'b h w -> (b h w)').cpu().numpy()))
        y_true = np.concatenate((y_true, rearrange(gt_masks, 'b c h w -> (b h w c)').cpu().numpy()))

        if CONFIG.visualize:
            unorm = transforms.Normalize((-1, -1,-1), (2, 2, 2))
            max_score = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var']) \
            + (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var']).item()
            img_in = rearrange(unorm(inputs), 'b c h w -> b h w c').cpu()
            gt_in = rearrange(gt_masks, 'b c h w -> b h w c').cpu()
            
            for b in range(inputs.shape[0]):                
                save_path = os.path.join(result_dir, f"images/img_{b:02d}{img_count}.png")
                visualize(img_in[b].squeeze(),
                          gt_in[b].squeeze(),
                          score_map[b].squeeze().cpu().numpy(),
                          max_score,
                          save_path=save_path)
                img_count += 1
                
    y_true = np.where(y_true > 0.5, 1, 0).astype(int)
    
    if len(np.unique(y_true) > 1):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        pixel_auroc = auc(fpr, tpr)
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        
        # ROC Curve 시각화 및 저장
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {pixel_auroc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {CONFIG.category}')
        plt.legend(loc="lower right")
        plot_path = os.path.join(result_dir, 'roc_curve.png')
        plt.savefig(plot_path)
        print(f"ROC curve plot saved to {plot_path}")
        plt.close()

    else:
        print("Could not calculate AUROC. All ground truth labels are the same.")
        
    # 점수 맵(y_score)과 정답 마스크(y_true)를 .npy 파일로 저장
    np.save(os.path.join(result_dir, 'y_score.npy'), y_score)
    np.save(os.path.join(result_dir, 'y_true.npy'), y_true)

    print("Results saved successfully.")

if __name__ == "__main__":
    detect_anomaly()

