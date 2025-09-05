import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class AnomalyDataset(Dataset):
    def __init__(self, root_dir, transforms=False, gt_transforms=False, split='train', category=None):
        super(AnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transforms = transforms
        self.gt_transforms = gt_transforms
        self.csv = self._get_datafile()

    def _get_datafile(self):
        if self.split == 'gt':
            data_path = os.path.join(self.root_dir, 'data', 'csv', 'ground_truth' + '.csv')
        else:
            data_path = os.path.join(self.root_dir, 'data', 'csv', self.split + '.csv')
        df = pd.read_csv(data_path)
        df = df.loc[df['category'] == self.category]
        return df

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sep = f'data{os.path.sep}mvtec_AD{os.path.sep}'
        row = self.csv.iloc[idx]

        if self.split == 'train':
            img_path = os.path.join(self.root_dir, sep, self.category, self.split, 'good',
                                    row['filename'])
            label = 0
        elif self.split == 'test':
            img_path = os.path.join(self.root_dir, sep, self.category, 'test', row['anomaly'],
                                    row['filename'])
            label = 1 if row['anomaly'] != 'good' else 0
        elif self.split == 'gt':
            img_path = os.path.join(self.root_dir, sep, self.category, 'ground_truth', row['anomaly'],
                                    row[idx]['filename'])
            label = 1 if row['anomaly'] != 'good' else 0

        else:
            raise ValueError(f'Unknown split: {self.split}')

        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        elif self.gt_transforms:
            img = self.gt_transforms(img)

        return img, label


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

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    # dataset = AnomalyDataset(root_dir=root_dir, category='zipper', transforms=transform, split='test')
    # dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    #
    # print(f'dataset length: {len(dataset)}')
    # print(f'dataloader length: {len(dataloader)}')
    # sep = os.path.sep
    #
    # for i, (img, label) in enumerate(dataloader):
    #     # save sample
    #     torchvision.utils.save_image(img, f'.{sep}samples{sep}sample_{i}.png', nrow=3)
    #
    #     if i == 10:
    #         print(img.size())
    #         print(label)
    #         break

    # dataset = AnomalyResnetDataset(root_dir=root_dir, category='zipper', transforms=transform)
    # dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    # origin_csv = pd.read_csv(r'C:\works\student-teacher\data\csv\resnet_train.csv')
    # print("origin csv length:", len(origin_csv.loc[origin_csv['category'] == 'zipper']))
    #
    # print(f'dataset length: {len(dataset)}')
    # print(f'dataloader length: {len(dataloader)}')
    # sep = os.path.sep
    #
    # for i, (img, label) in enumerate(dataloader):
    #     # save sample
    #     torchvision.utils.save_image(img, f'.{sep}samples{sep}sample_{i}.png', nrow=3)
    #
    #     if i == 10:
    #         print(img.size())
    #         print(label)
    #         break

    dataset = AnomalyDataset(root_dir=root_dir, category='zipper',
                             transforms=transform,
                             gt_transforms=transform,
                             split='gt')

    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    for i, (img, label) in enumerate(dataloader):
        # save sample
        torchvision.utils.save_image(img, f'.{sep}samples{sep}sample_{i}.png', nrow=3)

        if i == 10:
            print(img.size())
            print(label)
            break