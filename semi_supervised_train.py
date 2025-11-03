import torch
from torch.utils.data import DataLoader
from model import CrowdCCT
from enhanced_dataset import MixedCrowdDataset, DualDataset
from pseudo_labeler import PseudoLabeler
import torchvision.transforms as T
from utils import save_checkpoint
import numpy as np
import os

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

# ============================================
# CONFIGURATION
# ============================================
RESUME_TRAINING = True  # Set to False to start from scratch
CHECKPOINT_PATH = 'outputs/best_model.pth'
TOTAL_EPOCHS = 100  # Total epochs to train
FINE_TUNE_LR = 5e-6  # Lower LR when resuming (original was 1e-5)

# Data augmentation for semi-supervised learning
train_transform = T.Compose([
    T.Resize((384, 384)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor()
])

weak_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor()
])

strong_transform = T.Compose([
    T.Resize((384, 384)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomRotation(10),
    T.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize datasets
print("Loading labeled dataset...")
labeled_dataset = MixedCrowdDataset(
    'data/train/images',
    'data/train/annots',
    transform=train_transform,
    unlabeled=False
)

print("Loading unlabeled dataset...")
unlabeled_dataset = MixedCrowdDataset(
    'data/unlabeled/images',  # Put your unlabeled images here
    transform=train_transform,
    unlabeled=True
)

# Load or generate pseudo-labels
pseudo_label_path = 'outputs/pseudo_labels.json'
if os.path.exists(pseudo_label_path):
    print("Loading existing pseudo-labels...")
    unlabeled_dataset.load_pseudo_labels(pseudo_label_path)
else:
    print("Generating pseudo-labels for unlabeled data...")
    model = CrowdCCT().to(device)
    
    # Try to load pre-trained weights if available
    if os.path.exists('outputs/best_model.pth'):
        checkpoint = torch.load('outputs/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained model for pseudo-labeling")
    
    pseudo_labeler = PseudoLabeler(crowd_model=model, device=device)
    pseudo_labeler.label_dataset(unlabeled_dataset, method='ensemble', min_confidence=0.4)
    unlabeled_dataset.save_pseudo_labels(pseudo_label_path)

# Create combined dataset
print(f"Labeled samples: {len(labeled_dataset)}")
print(f"Unlabeled samples with pseudo-labels: {len(unlabeled_dataset.pseudo_labels)}")

# DataLoaders
labeled_loader = DataLoader(labeled_dataset, batch_size=8, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True)

# Initialize model and optimizer
model = CrowdCCT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = torch.nn.L1Loss()

# ============================================
# RESUME TRAINING FROM CHECKPOINT
# ============================================
start_epoch = 0
best_loss = np.inf

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print("\n" + "=" * 50)
    print(f"📦 Loading checkpoint from {CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Update learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = FINE_TUNE_LR
        
        # Get previous training info
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', np.inf)
        
        print(f"✅ Resumed from epoch {start_epoch}")
        print(f"   Previous best loss: {best_loss:.4f}")
        print(f"   New learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        print("Starting training from scratch...\n")
        start_epoch = 0
        best_loss = np.inf
else:
    if RESUME_TRAINING:
        print(f"\n⚠️  Checkpoint not found: {CHECKPOINT_PATH}")
    print("🆕 Starting training from scratch...\n")

# Semi-supervised training parameters
lambda_u = 0.5  # Weight for unlabeled loss
patience = 10
patience_counter = 0

print("\nStarting semi-supervised training...")
print("=" * 50)
print(f"Training from epoch {start_epoch} to {TOTAL_EPOCHS}")
print("=" * 50 + "\n")

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    labeled_loss_sum = 0.0
    unlabeled_loss_sum = 0.0
    total_samples = 0
    
    # Train on labeled data
    for imgs, targets, is_labeled in labeled_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        
        preds = model(imgs)
        targets = targets.view(-1)
        loss = criterion(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        labeled_loss_sum += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
    
    # Train on unlabeled data with pseudo-labels
    unlabeled_samples = 0
    for batch in unlabeled_loader:
        if len(batch) == 4:  # Has pseudo-label
            imgs, targets, is_labeled, confidence = batch
            imgs, targets = imgs.to(device), targets.to(device)
            
            # Only use samples with valid pseudo-labels
            valid_mask = targets >= 0
            if valid_mask.sum() == 0:
                continue
            
            imgs = imgs[valid_mask]
            targets = targets[valid_mask]
            confidence = confidence[valid_mask].to(device)
            
            preds = model(imgs)
            targets = targets.view(-1)
            
            # Weighted loss by confidence
            loss = (criterion(preds, targets) * confidence).mean()
            loss = loss * lambda_u  # Scale unlabeled loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            unlabeled_loss_sum += loss.item() * imgs.size(0)
            unlabeled_samples += imgs.size(0)
    
    # Calculate average losses
    avg_labeled_loss = labeled_loss_sum / total_samples if total_samples > 0 else 0
    avg_unlabeled_loss = unlabeled_loss_sum / unlabeled_samples if unlabeled_samples > 0 else 0
    total_loss = avg_labeled_loss + avg_unlabeled_loss
    
    # Early stopping and checkpointing
    if total_loss < best_loss:
        improvement = best_loss - total_loss
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}: Loss improved by {improvement:.4f}")
        print(f"  {best_loss:.4f} → {total_loss:.4f}")
        print(f"  Labeled: {avg_labeled_loss:.4f} | Unlabeled: {avg_unlabeled_loss:.4f}")
        
        best_loss = total_loss
        patience_counter = 0
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, total_loss, "outputs/best_model.pth")
        
        # Save backup every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, total_loss, f"outputs/checkpoint_epoch_{epoch+1}.pth")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}: Loss {total_loss:.4f} (no improvement, patience: {patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print("\n⏹️  Early stopping triggered!")
            break
    
    # Regenerate pseudo-labels every 10 epochs
    if (epoch + 1) % 10 == 0 and epoch > 0:
        print("\n🔄 Regenerating pseudo-labels with improved model...")
        pseudo_labeler = PseudoLabeler(crowd_model=model, device=device)
        pseudo_labeler.label_dataset(unlabeled_dataset, method='ensemble', min_confidence=0.4)
        unlabeled_dataset.save_pseudo_labels(pseudo_label_path)
        print("✅ Pseudo-labels updated\n")

print("\n" + "=" * 50)
print("TRAINING COMPLETED!")
print(f"Started from: Epoch {start_epoch}")
print(f"Final epoch: {epoch+1}")
print(f"Best loss: {best_loss:.4f}")
print("=" * 50)