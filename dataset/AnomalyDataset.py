import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class AnomalyDataset(Dataset):
    def __init__(self, root_dir, transforms=False, split='train', category=None):
        super(AnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transforms = transforms
        self.csv = self._get_datafile()

    def _get_datafile(self):
        data_path = os.path.join(self.root_dir, 'data', 'csv', self.split + '.csv')
        df = pd.read_csv(data_path)
        df = df.loc[df['category'] == self.category]
        return df

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]

        if self.split == 'train':
            img_path = os.path.join(self.root_dir, 'data/mvtec_AD/', self.category, self.split, 'good',
                                row['filename'])
            label = 0
        elif self.split == 'test':
            img_path = os.path.join(self.root_dir, 'data/mvtec_AD/', self.category, 'test', row['anomaly'],
                                    row['filename'])
            label = 1 if row['anomaly'] != 'good' else 0
        elif self.split == 'gt':
            img_path = os.path.join(self.root_dir, 'data/mvtec_AD', self.category, 'ground_truth', row[idx]['anomaly'],
                                    row[idx]['filename'])
            label = 1 if row['anomaly'] != 'good' else 0

        else:
            raise ValueError(f'Unknown split: {self.split}')

        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        return img, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test code
    from torchvision import transforms
    import torchvision

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    dataset = AnomalyDataset(root_dir=root_dir, category='zipper', transforms=transform, split='test')
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    print(f'dataset length: {len(dataset)}')
    print(f'dataloader length: {len(dataloader)}')

    for i, (img, label) in enumerate(dataloader):
        # save sample
        torchvision.utils.save_image(img, f'./samples/sample_{i}.png', nrow=3)

        if i == 10:
            print(img.size())
            print(label)
            break


