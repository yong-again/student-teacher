import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import os

from models.AnomalyNet import AnomalyNet
from dataset.AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var, load_model, get_model_path, create_experiment_path
from sklearn.metrics import roc_curve, auc
from configs import InferenceMultiScale


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
    t_mu, t_var, t_N = 0, 0, 0
    # Use a subset for faster calibration if the dataset is large
    calib_iter = min(len(dataloader), 50)
    for i, (images, _, _) in enumerate(dataloader):
        if i >= calib_iter:
            break
        inputs = images.to(device)
        t_out = teacher.fdfe(inputs)
        t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)

    max_err, max_var = 0, 0
    mu_err, var_err, N_err = 0, 0, 0
    mu_var, var_var, N_var = 0, 0, 0

    for i, (images, _, _) in enumerate(dataloader):
        if i >= calib_iter:
            break
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

def visualize(img, gt, score_map, save_path=None):
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
    im = plt.imshow(score_map, cmap='jet', vmin=0)
    plt.colorbar(im)
    plt.title("Anomaly Score Map")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def detect_anomaly_multiscale():
    CONFIG = InferenceMultiScale()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Runing multi-scale inference for {CONFIG.category} with patch sizes: {CONFIG.patch_sizes}")

    image_transform = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor()
    ])

    # --------- model load and calibration --------- #
    models_and_params_per_scale = {}


    # load calibration dataset
    calib_dataset = AnomalyDataset(
        root_dir=CONFIG.root_dir,
        category=CONFIG.category,
        split='train',
        transform=image_transform,
        mask_transform=mask_transform,
    )

    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    for p_size in CONFIG.patch_sizes:
        print(f"\n--- Loading models and calibrating for patch_size = {p_size} ---")

        teacher = AnomalyNet.create((p_size, p_size)).eval().to(device)
        teacher_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'teacher', p_size)
        load_model(teacher, teacher_model_path)

        # student networks
        students = [AnomalyNet.create((p_size, p_size)).eval().to(device) for _ in range(CONFIG.n_students)]
        for i, student in enumerate(students):
            student_model_path = get_model_path(CONFIG.root_dir, CONFIG.category, 'student', p_size, i)
            load_model(student, student_model_path)

        # calculate calibration parameters each scale
        params = calibrate(teacher, students, calib_dataloader, device)
        models_and_params_per_scale[p_size]  = {'teacher': teacher, 'students': students, 'params': params}

    # --------- inference --------- #
    test_dataset = AnomalyDataset(
        root_dir=CONFIG.root_dir,
        category=CONFIG.category,
        split='test',
        transform=image_transform,
        mask_transform=mask_transform
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    y_score, y_true = np.array([]), np.array([])
    result_dir = create_experiment_path(CONFIG.root_dir, category=f"{CONFIG.category}_multiscale")
    print(f"\nSaving results to {result_dir}")

    img_count = 0
    for images, labels, masks in tqdm(test_dataloader, desc="Multi-scale Inference"):
        inputs = images.to(device)
        gt_masks = masks.to(device)

        score_map_per_scale = []

        # get score map for each scale
        for p_size in CONFIG.patch_sizes:
            model_set = models_and_params_per_scale[p_size]
            teacher = model_set['teacher']
            students = model_set['students']
            params = model_set['params']

            score_map = get_score_map(inputs, teacher, students, params)
            score_map_per_scale.append(score_map)

    # aggregate score maps from different patch sizes
    final_score_map = torch.mean(torch.stack(score_map_per_scale, dim=0), dim=0)

    y_score = np.concatenate((y_score, rearrange(final_score_map, 'b h w -> (b h w)').cpu().numpy()))
    y_true = np.concatenate(((y_true, rearrange(gt_masks, 'b c h w -> (b h w c)').cpu().numpy())))

    if CONFIG.visualize:
        unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
        img_in = rearrange(unorm(inputs), 'b c h w -> b h w c').cpu()
        gt_in = rearrange(gt_masks, 'b c h w -> b h w c').cpu()


        for b in range(inputs.shape[0]):
            result_save_path = os.path.join(result_dir, "images")
            os.makedirs(result_save_path, exist_ok=True)
            save_path = os.path.join(result_save_path, f"img_{img_count:03d}.png")
            visualize(
                img_in[b].squeeze(),
                gt_in[b].squeeze(),
                final_score_map[b].squeeze().cpu().numpy(),
                save_path=save_path
            )
            img_count += 1

    y_true = np.where(y_true > 0.5, 1, 0).astype(int)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        pixel_auroc = auc(fpr, tpr)
        print(f"pixel-level AUROC: {pixel_auroc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {pixel_auroc:.4f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"Receiver Operating Characteristic for {CONFIG.category} (Multi-scale)")
        plt.legend(loc="lower right")
        plot_path = os.path.join(result_dir, f"roc_curve_{CONFIG.category}_multiscale.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    else:
        print("Only one class present in y_true. AUROC is not defined in this case.")

    np.save(os.path.join(result_dir, "y_score.npy"), y_score)
    np.save(os.path.join(result_dir, "y_true.npy"), y_true)
    print("Results saved successfully.")


if __name__ == "__main__":
    detect_anomaly_multiscale()



