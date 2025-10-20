# data/ade20k.py
import os, numpy as np, torch
from PIL import Image
from .base import BaseDataset

class ADE20KSegmentation(BaseDataset):
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 150

    def __init__(self, root, split='validation', mode='test',
                 transform=None, target_transform=None,
                 base_size=520, crop_size=480):
        super().__init__(root, split, mode, transform,
                         target_transform, base_size, crop_size)
        #data_root = os.path.join(root, self.BASE_DIR)
        data_root = root
        img_folder = os.path.join(data_root, 'images/validation')
        mask_folder = os.path.join(data_root, 'annotations/validation')
        self.images = sorted([
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder) if f.endswith('.jpg')
        ])
        self.masks = [
            os.path.join(mask_folder, os.path.basename(f).replace('.jpg','.png'))
            for f in self.images
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        mask = Image.open(self.masks[i])
        img, mask = self._val_sync_transform(img, mask)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        return img, mask, os.path.basename(self.images[i])

    @property
    def pred_offset(self):
        return 1
