import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def plot_predictions(image, pred_count, gt_count, save_path):
    plt.imshow(image)
    plt.title(f'Pred: {pred_count:.1f}  |  GT: {gt_count:.1f}')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
