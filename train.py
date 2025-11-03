import torch
from torch.utils.data import DataLoader
from model import CrowdCCT
from dataset import CrowdDataset
import torchvision.transforms as T
from utils import save_checkpoint
import numpy as np
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

train_transform = T.Compose([T.Resize((384,384)), T.RandomHorizontalFlip(), T.ToTensor()])
# Assuming 'data/train/images' and 'data/train/annots' exist
train_set = CrowdDataset('data/train/images', 'data/train/annots', transform=train_transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdCCT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = torch.nn.L1Loss() # MAE is a standard loss for crowd counting

# Initialize tracking for the best model
best_loss = np.inf

for epoch in range(50):
    model.train()
    total_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        
        # Ensure targets is the correct shape (B,) for L1Loss with model output (B,)
        targets = targets.view(-1)
        loss = criterion(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0) # Accumulate weighted loss
    
    avg_train_loss = total_loss / len(train_set)
    
    # Checkpoint saving logic: save only if current average loss is better than best_loss
    if avg_train_loss < best_loss:
        print(f"Epoch {epoch} loss improved from {best_loss:.4f} to {avg_train_loss:.4f}. Saving checkpoint...")
        best_loss = avg_train_loss
        save_checkpoint(model, optimizer, epoch, avg_train_loss, "outputs/best_model.pth")
    else:
        print(f"Epoch {epoch} loss {avg_train_loss:.4f} did not improve over {best_loss:.4f}.")