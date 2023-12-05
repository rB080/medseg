
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np


class Refuge_Dataset(Dataset):
    def __init__(self, root_path, split='train', image_size=256, mask_type='disc'):
        super().__init__()
        self.root_path = root_path
        self.split = split
        self.img_size = image_size
        self.mask_type = mask_type
        self.load_metadata(split=split)

    def load_image(self, path, is_mask=False):
        
        if not is_mask: img = cv2.imread(path)
        else: img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #assert len(self.img_size) != 2, f"Image size cannot be: {self.img_size}"
        img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

        img = np.array(img / 255.0, dtype=np.float32)  # normalization

        if is_mask:
            img = (img - img.min()) / img.max()
            img = 1 - img
            if self.mask_type == 'disc':
                img[img > 0.3] = 1.0
                img[img <= 0.3] = 0.0
            elif self.mask_type == 'cup':
                img[img > 0.7] = 1.0
                img[img <= 0.7] = 0.0

        return img
    
    def load_metadata(self, split='train'):
        img_dir = os.path.join(self.root_path, split, "images")
        gt_dir = os.path.join(self.root_path, split, "gts")
        data_list = sorted(os.listdir(img_dir))
        self.metadata = []
        for data in data_list:
            img_path = os.path.join(img_dir, data)
            gt_path = os.path.join(gt_dir, data[:-4]+".bmp")
            self.metadata.append((img_path, gt_path))
        
    def __getitem__(self, index):
        img_path, gt_path = self.metadata[index]
        
        img, mask = self.load_image(img_path), self.load_image(gt_path, is_mask=True)
        img, mask = torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        mask = mask.unsqueeze(0)

        return img, mask
    
    def __len__(self):
        return len(self.metadata)


class Isic_Dataset(Dataset):
    def __init__(self, root_path, split='train', image_size=256):
        super().__init__()
        self.root_path = root_path
        self.split = split
        self.img_size = image_size
        self.load_metadata(split=split)

    def load_image(self, path, is_mask=False):
        
        if not is_mask: img = cv2.imread(path)
        else: img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #assert len(self.img_size) != 2, f"Image size cannot be: {self.img_size}"
        img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

        img = np.array(img / 255.0, dtype=np.float32)  # normalization

        if is_mask:
            img = (img - img.min()) / img.max()
            

        return img
    
    def load_metadata(self, split='train'):
        if split == 'train': 
            img_directory = 'ISIC-2017_Training_Data'
            msk_directory = 'ISIC-2017_Training_Part1_GroundTruth'
        elif split == 'val': 
            img_directory = 'ISIC-2017_Validation_Data'
            msk_directory = 'ISIC-2017_Validation_Part1_GroundTruth'
        else:
            img_directory = 'ISIC-2017_Test_v2_Data'
            msk_directory = 'ISIC-2017_Test_v2_Part1_GroundTruth'

        img_dir = os.path.join(self.root_path, img_directory)
        gt_dir = os.path.join(self.root_path, msk_directory)
        data_list = sorted(os.listdir(gt_dir))
        self.metadata = []
        for data in data_list:
            img_path = os.path.join(img_dir, data[:-17]+".jpg")
            gt_path = os.path.join(gt_dir, data)
            self.metadata.append((img_path, gt_path))
        
    def __getitem__(self, index):
        img_path, gt_path = self.metadata[index]
        
        img, mask = self.load_image(img_path), self.load_image(gt_path, is_mask=True)
        img, mask = torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        mask = mask.unsqueeze(0)

        return img, mask
    
    def __len__(self):
        return len(self.metadata)