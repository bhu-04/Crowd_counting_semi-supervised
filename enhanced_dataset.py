import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from scipy.io import loadmat
import json

class MixedCrowdDataset(Dataset):
    """
    Dataset that supports both labeled and unlabeled data.
    - Labeled: .mat files with ground truth counts
    - Unlabeled: images without annotations (for pseudo-labeling)
    """
    def __init__(self, img_dir, ann_dir=None, transform=None, unlabeled=False):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.unlabeled = unlabeled
        
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not unlabeled and ann_dir:
            self.ann_files = sorted(os.listdir(ann_dir))
            # Ensure alignment
            assert len(self.img_files) == len(self.ann_files), \
                "Mismatch between images and annotations"
        else:
            self.ann_files = None
            
        # For storing pseudo-labels
        self.pseudo_labels = {}
        self.confidence_scores = {}
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
            
        # Return labeled data
        if not self.unlabeled and self.ann_files:
            ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
            mat = loadmat(ann_path)
            try:
                count = len(mat['image_info'][0,0][0,0][0])
            except:
                # Alternative key structure
                count = len(mat['annPoints']) if 'annPoints' in mat else 0
            return img_tensor, torch.tensor(count, dtype=torch.float32), True
        
        # Return unlabeled data (with pseudo-label if available)
        img_name = self.img_files[idx]
        if img_name in self.pseudo_labels:
            count = self.pseudo_labels[img_name]
            confidence = self.confidence_scores.get(img_name, 0.5)
            return img_tensor, torch.tensor(count, dtype=torch.float32), False, confidence
        else:
            return img_tensor, torch.tensor(-1.0, dtype=torch.float32), False, 0.0
    
    def add_pseudo_label(self, idx, count, confidence=1.0):
        """Add pseudo-label for an unlabeled image"""
        img_name = self.img_files[idx]
        self.pseudo_labels[img_name] = count
        self.confidence_scores[img_name] = confidence
    
    def save_pseudo_labels(self, path):
        """Save pseudo-labels to JSON file"""
        data = {
            'pseudo_labels': self.pseudo_labels,
            'confidence_scores': self.confidence_scores
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_pseudo_labels(self, path):
        """Load pseudo-labels from JSON file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.pseudo_labels = data.get('pseudo_labels', {})
                self.confidence_scores = data.get('confidence_scores', {})
            print(f"Loaded {len(self.pseudo_labels)} pseudo-labels")


class DualDataset(Dataset):
    """
    Combines labeled and unlabeled datasets for semi-supervised learning
    """
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_size = len(labeled_dataset)
        self.unlabeled_size = len(unlabeled_dataset)
        
    def __len__(self):
        return self.labeled_size + self.unlabeled_size
    
    def __getitem__(self, idx):
        if idx < self.labeled_size:
            return self.labeled_dataset[idx]
        else:
            return self.unlabeled_dataset[idx - self.labeled_size]
