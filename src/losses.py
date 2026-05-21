# Global hybrid loss functions (BCE + Dice)
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # Expects raw logits, applies sigmoid internally

    def forward(self, pred, target):
        # Calculate BCE with logits (numerically stable, AMP-compatible)
        bce_loss = self.bce(pred, target)
        
        # Apply sigmoid to predictions for Dice computation (needs probabilities)
        pred_prob = torch.sigmoid(pred)
        
        # Calculate Dice Loss on probabilities
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        # Combine equally
        return bce_loss + dice_loss