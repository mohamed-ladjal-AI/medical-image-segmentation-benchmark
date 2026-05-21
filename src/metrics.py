"""Evaluation metrics for 2D medical image segmentation.

This module implements a publication-grade metric suite including:
- Overlap: Dice (DSC) and IoU
- Boundary: 95% Hausdorff Distance (HD95) and Normalized Surface Dice (NSD)
- Clinical: False Positive Area (FPA) and Plaque Area Quantification Error

Design notes:
- All metrics are computed per-image (subject-wise) and returned as lists
  so callers can compute mean/median/std from those lists.
- Empty ground-truth frames (no plaque) are excluded from Dice/IoU/HD95
  aggregation pools and are tracked via FPA instead.
"""

from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion


def _binarize(tensor: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    # Apply sigmoid to convert logits to probabilities for threshold-based binarization
    tensor = torch.sigmoid(tensor)
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:  # [B, C, H, W]
        arr = arr[:, 0]
    return (arr > thr).astype(np.uint8)


def _surface_coords(mask: np.ndarray) -> np.ndarray:
    """Return coordinates of the object surface pixels (row, col).

    Uses a 3x3 erosion to compute the binary boundary (surface).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    boundary = mask ^ eroded
    coords = np.column_stack(np.nonzero(boundary))
    return coords


def _hd95_from_surfaces(a_surf: np.ndarray, b_surf: np.ndarray) -> float:
    """Compute the 95th-percentile symmetric Hausdorff distance between surfaces.

    Returns np.inf when one of the surfaces is empty and the other is not.
    Returns 0.0 when both surfaces are empty.
    """
    if a_surf.size == 0 and b_surf.size == 0:
        return 0.0
    if a_surf.size == 0 or b_surf.size == 0:
        return float('inf')

    tree_a = cKDTree(a_surf)
    tree_b = cKDTree(b_surf)

    dists_a_to_b, _ = tree_b.query(a_surf, k=1)
    dists_b_to_a, _ = tree_a.query(b_surf, k=1)

    all_dists = np.concatenate([dists_a_to_b, dists_b_to_a])
    return float(np.percentile(all_dists, 95))


def _nsd_from_surfaces(a_surf: np.ndarray, b_surf: np.ndarray, tol: float) -> float:
    """Compute Normalized Surface Dice (NSD) at tolerance `tol` (pixels).

    NSD = (|{s_a : d(s_a, S_b) <= tol}| + |{s_b : d(s_b, S_a) <= tol}|)
          / (|S_a| + |S_b|)
    """
    # If both empty -> perfect overlap
    if a_surf.size == 0 and b_surf.size == 0:
        return 1.0

    # handle empty surfaces
    if a_surf.size == 0:
        # proportion of b surface within tol of empty a is 0
        return 0.0 if b_surf.size > 0 else 1.0
    if b_surf.size == 0:
        return 0.0 if a_surf.size > 0 else 1.0

    tree_b = cKDTree(b_surf)
    tree_a = cKDTree(a_surf)

    dists_a_to_b, _ = tree_b.query(a_surf, k=1)
    dists_b_to_a, _ = tree_a.query(b_surf, k=1)

    a_within = np.count_nonzero(dists_a_to_b <= tol)
    b_within = np.count_nonzero(dists_b_to_a <= tol)

    denom = a_surf.shape[0] + b_surf.shape[0]
    if denom == 0:
        return 1.0
    return float((a_within + b_within) / denom)


def _dice_from_masks(p: np.ndarray, t: np.ndarray) -> float:
    inter = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _iou_from_masks(p: np.ndarray, t: np.ndarray) -> float:
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_to_mm2_ratio: float = 1.0,
    nsd_tolerance: float = 1.0,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute per-image metrics and aggregated statistics for a batch.

    Args:
        pred: torch.Tensor predictions (B,1,H,W) or (B,H,W)
        target: torch.Tensor ground truth masks (B,1,H,W) or (B,H,W)
        pixel_to_mm2_ratio: multiplier to convert pixel counts to mm^2
        nsd_tolerance: pixel tolerance for NSD (default 1.0)

    Returns:
        per_image_metrics: list of dicts, one per image, keys =
            {'dice','iou','hd95','nsd','fp_area','plaque_area_error','has_plaque'}
        aggregates: dict mapping metric -> {'mean','median','std','count'}
    """
    p_bin = _binarize(pred)
    t_bin = _binarize(target)

    batch_size = p_bin.shape[0]
    per_image = []

    for i in range(batch_size):
        p = p_bin[i]
        t = t_bin[i]

        has_plaque = bool(t.sum() > 0)

        # Areas
        pred_area_px = int(p.sum())
        true_area_px = int(t.sum())
        fp_pixels = int(np.logical_and(p == 1, t == 0).sum())

        fp_area = float(fp_pixels * pixel_to_mm2_ratio)
        plaque_area_error = float(abs(pred_area_px - true_area_px) * pixel_to_mm2_ratio)

        # Default metric values (use NaN for excluded values)
        dice = np.nan
        iou = np.nan
        hd95 = np.nan
        nsd = np.nan

        if has_plaque:
            dice = _dice_from_masks(p, t)
            iou = _iou_from_masks(p, t)

            a_surf = _surface_coords(p)
            b_surf = _surface_coords(t)

            try:
                hd95 = _hd95_from_surfaces(a_surf, b_surf)
            except Exception:
                hd95 = float('nan')

            try:
                nsd = _nsd_from_surfaces(a_surf, b_surf, tol=nsd_tolerance)
            except Exception:
                nsd = float('nan')

        per_image.append(
            {
                'dice': float(dice) if not np.isnan(dice) else np.nan,
                'iou': float(iou) if not np.isnan(iou) else np.nan,
                'hd95': float(hd95) if not np.isnan(hd95) else np.nan,
                'nsd': float(nsd) if not np.isnan(nsd) else np.nan,
                'fp_area': float(fp_area),
                'plaque_area_error': float(plaque_area_error),
                'has_plaque': bool(has_plaque),
            }
        )

    # Aggregation helper
    def _agg(name: str, items: List[Dict[str, float]], exclude_empty_gt: bool = False):
        vals = []
        for it in items:
            if exclude_empty_gt and not it['has_plaque']:
                continue
            v = it.get(name, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals.append(v)
        if len(vals) == 0:
            return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'count': 0}
        arr = np.asarray(vals, dtype=float)
        return {'mean': float(arr.mean()), 'median': float(np.median(arr)), 'std': float(arr.std(ddof=0)), 'count': int(arr.size)}

    aggregates = {
        'dice': _agg('dice', per_image, exclude_empty_gt=True),
        'iou': _agg('iou', per_image, exclude_empty_gt=True),
        'hd95': _agg('hd95', per_image, exclude_empty_gt=True),
        'nsd': _agg('nsd', per_image, exclude_empty_gt=True),
        'fp_area': _agg('fp_area', per_image, exclude_empty_gt=False),
        'plaque_area_error': _agg('plaque_area_error', per_image, exclude_empty_gt=False),
    }

    return per_image, aggregates


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, pixel_to_mm2_ratio: float = 1.0):
    """Backward-compatible helper that returns per-image Dice list and FPA list.

    Prefer `evaluate_batch` for the full metric suite.
    """
    per_image, _ = evaluate_batch(pred, target, pixel_to_mm2_ratio=pixel_to_mm2_ratio)
    dice_list = [it['dice'] for it in per_image if it['has_plaque'] and not np.isnan(it['dice'])]
    fp_list = [it['fp_area'] for it in per_image if (not it['has_plaque']) or (it['fp_area'] > 0)]
    return dice_list, fp_list