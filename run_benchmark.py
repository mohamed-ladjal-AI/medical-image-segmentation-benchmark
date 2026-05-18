import argparse
import os
import torch
import json
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import numpy as np
from tqdm import tqdm

# Import local global assets
from src.dataset import CarotidPlaqueDataset
from src.losses import HybridLoss
from src.metrics import evaluate_batch          # ← updated import
from src.generate_plots import plot_training_curves
from src.repro import set_seed, seed_worker

def main():
    parser = argparse.ArgumentParser(description="Unified Carotid Plaque Segmentation Benchmark Engine")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["unet", "unet_plus_plus", "attention_unet", "deeplabv3_plus", "hrnet", "segformer", "my_network"],
                        help="Specify which architecture pipeline to execute.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory to store experiment artifacts.")
    parser.add_argument("--exp_name", type=str, default="", help="Optional experiment name to include in output path.")
    parser.add_argument("--hyperparameter", type=str, default="", help="Path to JSON file with tuned hyperparameters (e.g., pipelines/unet/config.json.)")
    parser.add_argument("--split_file", type=str, default="", help="Path to JSON split file to load/save train/val filenames. If omitted, uses dataset/splits/split_seed<seed>.json")
    parser.add_argument("--split_ratio", type=float, default=0.10, help="Fraction of training set to use for validation (e.g., 0.10 for 10%)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using Execution Device: {device}")

    # ---------- Reproducibility ----------
    set_seed(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_base = Path(args.output_dir)
    exp_id = f"{args.model}_{args.exp_name or timestamp}_seed{args.seed}_{timestamp}"
    exp_dir = exp_base / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save CLI args for reproducibility
    with open(exp_dir / 'run_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))

    # ==========================================
    # 1. FIXED PRE-PROCESSING & AUGMENTATIONS
    # ==========================================
    from src.data_augmentation import get_transforms

    train_transform, val_transform = get_transforms(input_size=512)

    # ==========================================
    # 2. STANDARDIZED DATA LOADING
    # ==========================================
    train_dir = Path("dataset/train")
    if not train_dir.exists():
        raise FileNotFoundError("dataset/train does not exist. Please provide training data in dataset/train")

    all_files = sorted([
        f for f in os.listdir(train_dir)
        if "_labeled" not in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if len(all_files) == 0:
        raise FileNotFoundError("No training images found in dataset/train")

    if args.split_file:
        split_path = Path(args.split_file)
    else:
        split_dir = Path("dataset/splits")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_dir / f"split_seed{args.seed}.json"

    if not (0.0 < args.split_ratio < 0.5):
        raise ValueError("--split_ratio must be between 0 and 0.5 (exclusive)")

    if split_path.exists():
        with open(split_path, 'r') as f:
            saved = json.load(f)
        train_files = saved.get('train', [])
        val_files = saved.get('val', [])
        print(f"🔁 Loaded existing train/val split from {split_path}")
    else:
        import random as _random
        rnd = _random.Random(args.seed)
        files = list(all_files)
        rnd.shuffle(files)
        val_count = max(1, int(len(files) * args.split_ratio))
        val_files = files[:val_count]
        train_files = files[val_count:]

        with open(split_path, 'w') as f:
            json.dump({'train': train_files, 'val': val_files}, f, indent=2)
        print(f"💾 Saved deterministic train/val split to {split_path}")

    train_dataset = CarotidPlaqueDataset(
        data_dir=str(train_dir), 
        transform=train_transform,
        seed=args.seed,
        filenames=train_files,
    )
    val_dataset = CarotidPlaqueDataset(
        data_dir=str(train_dir), 
        transform=val_transform,
        seed=args.seed,
        filenames=val_files,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker
    )

    # ==========================================
    # 3. LOAD HYPERPARAMETERS BEFORE MODEL INSTANTIATION
    # ==========================================
    hyperparameter_config = {}
    if args.hyperparameter:
        hyper_path = Path(args.hyperparameter)
        if not hyper_path.exists():
            raise FileNotFoundError(f"Hyperparameter config not found: {hyper_path}")
        with open(hyper_path, 'r') as f:
            hyperparameter_config = json.load(f)
        print(f"📋 Loaded tuned hyperparameters from: {hyper_path}")
    else:
        print(f"⚠️  No --hyperparameter provided; using defaults from get_model()")

    # ==========================================
    # 4. DYNAMIC PIPELINE LOADING
    # ==========================================
    print(f"📦 Loading architecture pipeline: {args.model}")
    
    if args.model == "unet":
        from pipelines.unet.model import get_model
    elif args.model == "unet_plus_plus":
        from pipelines.unet_plus_plus.model import get_model
    elif args.model == "attention_unet":
        from pipelines.attention_unet.model import get_model
    elif args.model == "deeplabv3_plus":
        from pipelines.deeplabv3_plus.model import get_model
    elif args.model == "hrnet":
        from pipelines.hrnet.model import get_model
    elif args.model == "segformer":
        from pipelines.segformer.model import get_model
    elif args.model == "my_network":
        from pipelines.my_network.model import get_model

    model, config = get_model(config_override=hyperparameter_config if hyperparameter_config else None)
    model = model.to(device)

    with open(exp_dir / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    criterion = HybridLoss()

    optimizer_name = config.get("optimizer_name", "AdamW")
    optimizer_map = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }
    
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizer_map.keys())}")
    
    optimizer_class = optimizer_map[optimizer_name]
    optimizer = optimizer_class(
        model.parameters(), 
        lr=config.get("lr", 1e-4), 
        weight_decay=config.get("weight_decay", 1e-5)
    )
    print(f"⚡ Using optimizer: {optimizer_name} (lr={config.get('lr', 1e-4)}, weight_decay={config.get('weight_decay', 1e-5)})")

    # ==========================================
    # 5. UNIFIED TRAINING & VALIDATION ENGINE
    # ==========================================
    best_val_dice = 0.0
    epoch_logs = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_hd95': [],
        'val_nsd': [],
        'val_fp_area': [],
        'val_plaque_area_err': [],
    }

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        # ── Training Cycle ────────────────────────────────────────────────
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ── Validation Cycle ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_per_image = []   # accumulates one dict per image across all batches

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

                # evaluate_batch returns (per_image_list, aggregates_dict)
                # We only need per_image here; epoch-level aggregation is done below.
                per_image, _ = evaluate_batch(outputs, masks)
                all_per_image.extend(per_image)

        # ── Epoch-level aggregation (subject-wise) ────────────────────────
        def _collect(key, plaque_only=False):
            vals = [
                m[key] for m in all_per_image
                if (not plaque_only or m['has_plaque'])
                and not np.isnan(m[key])
            ]
            return np.asarray(vals, dtype=float) if vals else np.array([np.nan])

        dice_vals  = _collect('dice',              plaque_only=True)
        iou_vals   = _collect('iou',               plaque_only=True)
        hd95_vals  = _collect('hd95',              plaque_only=True)
        nsd_vals   = _collect('nsd',               plaque_only=True)
        fp_vals    = _collect('fp_area',           plaque_only=False)
        pae_vals   = _collect('plaque_area_error', plaque_only=False)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        mean_dice   = float(np.nanmean(dice_vals))
        mean_iou    = float(np.nanmean(iou_vals))
        mean_hd95   = float(np.nanmean(hd95_vals))
        mean_nsd    = float(np.nanmean(nsd_vals))
        median_fp   = float(np.nanmedian(fp_vals))
        mean_pae    = float(np.nanmean(pae_vals))

        epoch_logs['train_loss'].append(avg_train_loss)
        epoch_logs['val_loss'].append(avg_val_loss)
        epoch_logs['val_dice'].append(mean_dice)
        epoch_logs['val_iou'].append(mean_iou)
        epoch_logs['val_hd95'].append(mean_hd95)
        epoch_logs['val_nsd'].append(mean_nsd)
        epoch_logs['val_fp_area'].append(median_fp)
        epoch_logs['val_plaque_area_err'].append(mean_pae)

        # ── Console summary ───────────────────────────────────────────────
        print(f"📊 Epoch {epoch+1} → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Overlap  — Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")
        print(f"   Boundary — HD95: {mean_hd95:.4f} px | NSD: {mean_nsd:.4f}")
        print(f"   Clinical — Median FP Area: {median_fp:.4f} mm² | Mean Plaque Area Error: {mean_pae:.4f} mm²\n")

        # ── TensorBoard ───────────────────────────────────────────────────
        writer.add_scalar('loss/train',                  avg_train_loss, epoch + 1)
        writer.add_scalar('loss/val',                    avg_val_loss,   epoch + 1)
        writer.add_scalar('metrics/mean_dice',           mean_dice,      epoch + 1)
        writer.add_scalar('metrics/mean_iou',            mean_iou,       epoch + 1)
        writer.add_scalar('metrics/mean_hd95',           mean_hd95,      epoch + 1)
        writer.add_scalar('metrics/mean_nsd',            mean_nsd,       epoch + 1)
        writer.add_scalar('metrics/median_fp_area',      median_fp,      epoch + 1)
        writer.add_scalar('metrics/mean_plaque_area_err',mean_pae,       epoch + 1)

        # ── Live plots (overwritten each epoch) ─────────────────────────
        plot_training_curves(args.model, epoch_logs, exp_dir / 'plots', all_per_image)

        # ── Checkpoint (best Dice) ────────────────────────────────────────
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            save_path = exp_dir / 'best_weights.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
            }, str(save_path))
            print(f"💾 New best model saved to {save_path}!\n")

    # ── Finalise ──────────────────────────────────────────────────────────
    writer.flush()
    writer.close()

    summary = {
        'best_val_dice': float(best_val_dice),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    }
    with open(exp_dir / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()