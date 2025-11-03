import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from scipy.io import loadmat

class CrowdDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.ann_files = sorted(os.listdir(ann_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
        img = Image.open(img_path).convert('RGB')
        mat = loadmat(ann_path)
        count = len(mat['image_info'][0,0][0,0][0])  # Adjust key if dataset differs!
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(count, dtype=torch.float32)
