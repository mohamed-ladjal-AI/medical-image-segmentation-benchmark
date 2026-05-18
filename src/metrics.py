# Evaluation metrics scripts (Dice, HD95, TPA Error)
import torch
import numpy as np

def calculate_metrics(pred, target, pixel_to_mm2_ratio=1.0):
    """
    Args:
        pred (torch.Tensor): Model predictions binarized (0 or 1), shape [B, 1, H, W]
        target (torch.Tensor): Ground truth binary mask, shape [B, 1, H, W]
    """
    pred_np = pred.detach().cpu().numpy() > 0.5
    target_np = target.detach().cpu().numpy() > 0.5
    
    batch_size = pred_np.shape[0]
    dice_list = []
    fp_area_list = []
    
    for i in range(batch_size):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        
        has_plaque = np.sum(t) > 0
        
        if has_plaque:
            # Metric for plaque images: Dice Similarity Coefficient
            intersection = np.sum(p & t)
            union = np.sum(p) + np.sum(t)
            dice = (2.0 * intersection) / (union) if union > 0 else 1.0
            dice_list.append(dice)
        else:
            # Metric for plaque-free images: False Positive Area
            # Count pixels falsely classified as plaque
            fp_pixels = np.sum((p == 1) & (t == 0))
            fp_area = fp_pixels * pixel_to_mm2_ratio
            fp_area_list.append(fp_area)
            
    return dice_list, fp_area_list