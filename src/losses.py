# Global hybrid loss functions (BCE + Dice)
import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # Ensure values are strictly clamped to prevent log(0) errors
        pred = torch.clamp(pred, min=self.smooth, max=1.0 - self.smooth)
        
        # Calculate BCE
        bce_loss = self.bce(pred, target)
        
        # Calculate Dice Loss
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        # Combine equally
        return bce_loss + dice_loss