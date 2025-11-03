import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

class PseudoLabeler:
    """
    Generates pseudo-labels for unlabeled crowd images using:
    1. Pre-trained object detection models (for sparse crowds)
    2. Trained CrowdCCT model (for dense crowds)
    3. Ensemble methods for better accuracy
    """
    def __init__(self, crowd_model=None, device='cuda'):
        self.device = device
        self.crowd_model = crowd_model
        
        # Load pre-trained Faster R-CNN for person detection
        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT'
        ).to(device).eval()
        
        self.transform = T.Compose([
            T.ToTensor()
        ])
        
        self.crowd_transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor()
        ])
    
    def detect_people(self, image_path, confidence_threshold=0.5):
        """Use Faster R-CNN to detect people in image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.detector(img_tensor)[0]
        
        # Filter for 'person' class (class 1 in COCO)
        person_mask = predictions['labels'] == 1
        scores = predictions['scores'][person_mask]
        high_conf_mask = scores > confidence_threshold
        
        return high_conf_mask.sum().item(), scores[high_conf_mask].mean().item() if high_conf_mask.any() else 0.0
    
    def predict_with_crowd_model(self, image_path):
        """Use trained CrowdCCT model for prediction"""
        if self.crowd_model is None:
            return None, 0.0
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.crowd_transform(img).unsqueeze(0).to(self.device)
        
        self.crowd_model.eval()
        with torch.no_grad():
            count = self.crowd_model(img_tensor).item()
        
        return max(0, count), 0.8  # Confidence based on model training
    
    def generate_pseudo_label(self, image_path, method='ensemble'):
        """
        Generate pseudo-label using specified method:
        - 'detection': Use object detection only
        - 'crowd_model': Use crowd counting model only
        - 'ensemble': Combine both methods
        """
        detector_count, detector_conf = self.detect_people(image_path)
        
        if method == 'detection':
            return detector_count, detector_conf
        
        crowd_count, crowd_conf = self.predict_with_crowd_model(image_path)
        
        if method == 'crowd_model' and crowd_count is not None:
            return crowd_count, crowd_conf
        
        # Ensemble method
        if crowd_count is None:
            return detector_count, detector_conf
        
        # Weighted average based on density
        # If detection finds many people (>20), trust detection more
        # If detection finds few people (<10), trust crowd model more
        if detector_count > 20:
            weight = 0.7
        elif detector_count < 10:
            weight = 0.3
        else:
            weight = 0.5
        
        final_count = weight * detector_count + (1 - weight) * crowd_count
        final_conf = weight * detector_conf + (1 - weight) * crowd_conf
        
        return round(final_count), final_conf
    
    def label_dataset(self, dataset, method='ensemble', min_confidence=0.3):
        """
        Generate pseudo-labels for entire unlabeled dataset
        Only assigns labels if confidence exceeds threshold
        """
        print(f"Generating pseudo-labels using {method} method...")
        labeled_count = 0
        
        for idx in tqdm(range(len(dataset))):
            img_path = os.path.join(dataset.img_dir, dataset.img_files[idx])
            
            try:
                count, confidence = self.generate_pseudo_label(img_path, method)
                
                if confidence >= min_confidence:
                    dataset.add_pseudo_label(idx, count, confidence)
                    labeled_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"Successfully generated {labeled_count}/{len(dataset)} pseudo-labels")
        return labeled_count
