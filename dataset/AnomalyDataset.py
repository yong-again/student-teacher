import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple


# class AnomalyDataset(Dataset):
#     def __init__(self, root_dir, transform=None, mask_transform=None, split='train', category=None):
#         super(AnomalyDataset, self).__init__()
#         self.root_dir = root_dir
#         self.category = category
#         self.split = split
#         self.transform = transform
#         self.mask_transform = mask_transform
#         self.csv = self._get_datafile()
#
#         if mask_transform is None:
#             self.mask_transform = transforms.Compose([transforms.ToTensor()])
#         else:
#             self.mask_transform = mask_transform
#
#         if self.split == 'test':
#             self.csv = self._get_datafile()
#             self.image_files = self.csv.apply(lambda row: os.path.join(self.root_dir,\
#                 f'data/mvtec_AD/{self.category}/test/{row["anomaly"]}/{row["filename"]}'), axis=1).tolist()
#         elif self.split == 'train':
#             train_dir = os.path.join(self.root_dir, f'data/mvtec_AD/{self.category}/train/good')
#             self.image_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
#         else:
#             raise ValueError(f"Unknown split: {self.split}")
#
#     def _get_datafile(self):
#         data_path = os.path.join(self.root_dir, 'data', 'csv', 'test.csv')
#         df = pd.read_csv(data_path)
#         df = df.loc[df['category'] == self.category]
#         df = df.loc[df['anomaly'] != 'good']
#
#         return df
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):#         mask: Image.Image
#         image_path = self.image_files[idx]
#         img = Image.open(image_path).convert('RGB')
#
#         if self.split == 'train':
#             label = 0
#         else:
#             parts = image_path.split(os.path.sep)
#             anomaly_type = parts[-2]
#             filename = parts[-1]
#             label = 1 if anomaly_type != 'good' else 0
#
#             if self.split == 'train':
#                 mask = Image.new('L', img.size, 0)
#
#             else:
#                 try:
#                     filename_without_ext = os.path.splitext(filename)[0]
#                     mask_filename = f"{filename_without_ext}_mask.png"
#                     mask_path = os.path.join(self.root_dir, f'data/mvtec_AD/{self.category}/ground_truth/{anomaly_type}/{mask_filename}')
#                     mask = Image.open(mask_path).convert('L')
#
#                 except FileNotFoundError:
#                     print(f"Mask not found for {image_path}, using empty mask.")
#                     mask = Image.new('L', img.size, 0)
#
#         if self.transform:
#             img = self.transform(img)
#
#         if self.mask_transform:
#             mask = self.mask_transform(mask)
#
#         return img, label, mask


class AnomalyDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 mask_transform: Optional[transforms.Compose] = None,
                 split: str = 'train',
                 category: Optional[str] = None,
                 include_good_in_test: bool = True,
                 target_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        assert split in ['train', 'test'], f"split must be 'train' or 'test', Unknown split: {split}"
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.include_good_in_test = include_good_in_test
        self.target_size = target_size

        if mask_transform is None:
            mtfs = []
            if target_size is not None:
                mtfs.append(transforms.Resize(target_size, interpolation=Image.NEAREST))
            mtfs.append(transforms.ToTensor())
            self.mask_transform = transforms.Compose(mtfs)
        else:
            self.mask_transform = mask_transform

        if self.split == 'test':
            self.csv = self._get_test_csv()
            if not self.include_good_in_test:
                self.csv = self.csv.loc[self.csv['anomaly'] != 'good']
            self.image_files = self.csv.apply(
                lambda row: os.path.join(self.root_dir, f'data/mvtec_AD/{self.category}/test/{row["anomaly"]}'),
                axis=1
            ).tolist()

        else:
            train_dir = os.path.join(self.root_dir, f'data/mvtec_AD/{self.category}/train/good')
            self.image_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
            self.image_files.sort()

    def _get_test_csv(self) -> pd.DataFrame:
        data_path = os.path.join(self.root_dir, 'data', 'csv', 'test.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found at {data_path}")
        df = pd.read_csv(data_path)
        df = df.loc[df['category'] == self.category]
        return df

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img = Image.open(image_path).convert('RGB')

        parts = image_path.split(os.path.sep)
        # .../train/good/xxx.png  or  .../test/<anomaly_type>/xxx.png
        anomaly_type = parts[-2]
        label = 0 if anomaly_type == 'good' else 1

        if self.split == 'train' or anomaly_type == 'good':
            mask = Image.new('L', img.size, 0)
        else:
            filename = parts[-1]
            filename_without_ext = os.path.splitext(filename)[0]
            make_filename = f"{filename_without_ext}_mask.png"
            maks_path = os.path.join(
                self.root_dir,
                f'data/mvtec_AD/{self.category}/ground_truth/{anomaly_type}/{make_filename}'
            )
            if os.path.exists(maks_path):
                mask = Image.open(maks_path).convert('L')
            else:
                print(f"Mask not found for {image_path}, using empty mask.")
                mask = Image.new('L', img.size, 0)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, label, mask


# ------------------
# Resnet Dataset
# ------------------
class AnomalyResnetDataset(Dataset):
    def __init__(self, root_dir, transforms=False, category=None):
        super(AnomalyResnetDataset, self).__init__()
        self.root_dir = root_dir
        self.category = category
        self.transforms = transforms
        self.csv = self._get_datafile()

    def _get_datafile(self):
        data_path = os.path.join(self.root_dir, 'data', 'csv', 'resnet_train.csv')
        df = pd.read_csv(data_path)
        df = df.loc[df['category'] == self.category]

        return df

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sep = f'data{os.path.sep}mvtec_AD{os.path.sep}'
        row = self.csv.iloc[idx]

        if row['split'] == 'train':
            img_path = os.path.join(self.root_dir, sep, self.category, row['split'], 'good',
                                    row['filename'])
            label = 1 if row['split'] != 'good' else 0
        elif row['split'] == 'test':
            img_path = os.path.join(self.root_dir, sep, self.category, row['split'], row['anomaly'],
                                    row['filename'])
            label = 1 if row['split'] != 'good' else 0
        else:
            raise ValueError(f'Unknown split: {self.split}')

        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        return img, label

if __name__ == '__main__':
    # test code
    from torchvision import transforms
    import torchvision
    from torch.utils.data import DataLoader, Dataset

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    train_dataset = AnomalyDataset(root_dir, transform=transform, split='train', category='bottle')
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=4
    )
    print("dataset length:", len(train_dataset))
    print("train_loader length:", len(train_dataloader))
    
    for i, (image, label, mask) in enumerate(train_dataloader):
        print(image.size)
        print(label.size)
        print(mask.size)
        
        break
