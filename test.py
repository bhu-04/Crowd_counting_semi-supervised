import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import torch
from PIL import Image
from torchvision import transforms

# === 1. Load and check training annotations (.mat files) ===
annots_dir = r"C:\crowd_count\data\train\annots"

train_counts = []

# Read all .mat annotation files
for annot_file in os.listdir(annots_dir):
    if annot_file.endswith('.mat'):
        annot_path = os.path.join(annots_dir, annot_file)
        
        try:
            mat_data = sio.loadmat(annot_path)
            
            # Common formats in crowd counting datasets:
            # Try different possible keys
            if 'annPoints' in mat_data:
                points = mat_data['annPoints']
                count = len(points) if len(points.shape) == 2 else points.shape[0]
            elif 'image_info' in mat_data:
                points = mat_data['image_info'][0][0][0][0][0]
                count = points.shape[0]
            elif 'annotation' in mat_data:
                points = mat_data['annotation']
                count = points.shape[0]
            elif 'points' in mat_data:
                points = mat_data['points']
                count = points.shape[0]
            else:
                # Print available keys to help debug
                print(f"Unknown format in {annot_file}. Keys: {mat_data.keys()}")
                continue
                
            train_counts.append(count)
            
        except Exception as e:
            print(f"Error reading {annot_file}: {e}")

if train_counts:
    plt.figure(figsize=(10, 6))
    plt.hist(train_counts, bins=50, edgecolor='black')
    plt.title("Training Data Count Distribution")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\n=== TRAINING DATA STATISTICS ===")
    print(f"Total training samples: {len(train_counts)}")
    print(f"Max training count: {max(train_counts)}")
    print(f"Min training count: {min(train_counts)}")
    print(f"Mean training count: {np.mean(train_counts):.2f}")
    print(f"Median training count: {np.median(train_counts):.2f}")
    print(f"75th percentile: {np.percentile(train_counts, 75):.2f}")
    print(f"95th percentile: {np.percentile(train_counts, 95):.2f}")
    
    # This is CRITICAL - if max is < 1000 and you're testing on 180K, that's your problem!
    if max(train_counts) < 1000:
        print("\n⚠️ WARNING: Your training data has very small crowds!")
        print(f"   Max training count is only {max(train_counts)}")
        print(f"   But you're testing on 180,000 people!")
        print(f"   Your model CANNOT generalize to such large crowds.")
else:
    print("❌ No counts found! Let me check the .mat file structure...")
    # Let's inspect the first .mat file
    first_mat = [f for f in os.listdir(annots_dir) if f.endswith('.mat')][0]
    mat_path = os.path.join(annots_dir, first_mat)
    mat_data = sio.loadmat(mat_path)
    print(f"\nInspecting {first_mat}:")
    print(f"Keys in .mat file: {mat_data.keys()}")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(mat_data[key])}, shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")

# === 2. Load model and test predictions ===
model_path = r"C:\crowd_count\outputs\best_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    checkpoint = torch.load(model_path, map_location=device)
    print(f"\n=== MODEL INFO ===")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Number of model layers: {len(state_dict)}")
    
    # Check if there's training info
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'best_mae' in checkpoint:
        print(f"Best MAE during training: {checkpoint['best_mae']:.2f}")
    
except Exception as e:
    print(f"Error loading model: {e}")

# === 3. Check test images ===
test_images_dir = r"C:\crowd_count\data\test\images"
test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"\n=== TEST DATA ===")
print(f"Found {len(test_images)} test images")
print(f"Sample images: {test_images[:3]}")