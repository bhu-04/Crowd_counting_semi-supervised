import torch
from torch.utils.data import DataLoader
from model import CrowdCCT
from dataset import CrowdDataset
from utils import plot_predictions, load_checkpoint 
import torchvision.transforms as T
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Create outputs directory for result images if it doesn't exist
os.makedirs("outputs", exist_ok=True)

test_transform = T.Compose([T.Resize((384,384)), T.ToTensor()])
# Assuming 'data/test/images' and 'data/test/annots' exist
test_set = CrowdDataset('data/test/images', 'data/test/annots', transform=test_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # Batch size is 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdCCT().to(device)

# Load the best model weights
try:
    # Creating a dummy optimizer to satisfy load_checkpoint signature, 
    # but only model_state_dict is needed for evaluation.
    optimizer_placeholder = torch.optim.Adam(model.parameters(), lr=1e-5) 
    epoch, loss = load_checkpoint(model, optimizer_placeholder, 'outputs/best_model.pth')
    print(f"Loaded model from epoch {epoch} with training loss {loss:.4f}")
except FileNotFoundError:
    print("Error: 'outputs/best_model.pth' not found. Please train the model first.")
    exit()

model.eval()
preds, gts = [], []
with torch.no_grad():
    for i, (imgs, targets) in enumerate(test_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        out = model(imgs)
        
        # --- FIX APPLIED HERE ---
        # Use .item() to extract the single scalar prediction as a float, 
        # preventing the 0-d array iteration error.
        preds.append(out.item())
        gts.append(targets.item())
        # ------------------------
        
        # Plotting the image (Batch size 1, so index 0 is correct)
        plot_predictions(imgs[0].permute(1,2,0).cpu().numpy(), preds[-1], gts[-1], f"outputs/result_{i}.png")
        
print("-" * 30)
print("Evaluation Results:")
print("MAE:", mean_absolute_error(gts, preds))
print("MSE:", mean_squared_error(gts, preds))
print("-" * 30)