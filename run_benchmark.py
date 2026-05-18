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
from src.metrics import calculate_metrics
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
    # Use centralized augmentation helper to keep run script small
    from src.data_augmentation import get_transforms

    train_transform, val_transform = get_transforms(input_size=512)

    # ==========================================
    # 2. STANDARDIZED DATA LOADING
    # ==========================================
    # If an explicit validation folder exists, use it. Otherwise split 10% from train.
    train_dir = Path("dataset/train")
    if not train_dir.exists():
        raise FileNotFoundError("dataset/train does not exist. Please provide training data in dataset/train")

    # No explicit validation folder requested; deterministically split from train and save/load the split
    all_files = sorted([
        f for f in os.listdir(train_dir)
        if "_labeled" not in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if len(all_files) == 0:
        raise FileNotFoundError("No training images found in dataset/train")

    # Determine canonical split file path
    if args.split_file:
        split_path = Path(args.split_file)
    else:
        split_dir = Path("dataset/splits")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_path = split_dir / f"split_seed{args.seed}.json"

    # Validate split ratio
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

    # Deterministic DataLoader generator
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
    # Load hyperparameters first so architecture decisions use correct config
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
    
    # Import the respective factory function dynamically based on runtime choices
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

    # Instantiate model with hyperparameter config passed in
    model, config = get_model(config_override=hyperparameter_config if hyperparameter_config else None)
    model = model.to(device)

    # Save model-specific config
    with open(exp_dir / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Standardized Loss Function
    criterion = HybridLoss()

    # Dynamic Optimization (using tuned hyperparameters, but structured identical framework)
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

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        # Training Cycle
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

        # Validation Cycle
        model.eval()
        val_loss = 0.0
        all_dice = []
        all_fp_areas = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Global metrics parser execution
                batch_dice, batch_fp = calculate_metrics(outputs, masks)
                all_dice.extend(batch_dice)
                all_fp_areas.extend(batch_fp)

        # Average results
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        mean_dice = np.mean(all_dice) if len(all_dice) > 0 else 0.0
        median_fp = np.median(all_fp_areas) if len(all_fp_areas) > 0 else 0.0

        print(f"📊 Summary Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   ✨ Plaque Images Mean Dice: {mean_dice:.4f} | Healthy Images Median FP Area: {median_fp:.4f} mm²\n")

        # Log to TensorBoard
        writer.add_scalar('loss/train', avg_train_loss, epoch+1)
        writer.add_scalar('loss/val', avg_val_loss, epoch+1)
        writer.add_scalar('metrics/mean_dice', float(mean_dice), epoch+1)
        writer.add_scalar('metrics/median_fp_area', float(median_fp), epoch+1)

        # Checkpoint Saving Layer
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            save_path = exp_dir / 'best_weights.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
            }, str(save_path))
            print(f"💾 New best model saved to {save_path}!")

    # Finalize
    writer.flush()
    writer.close()

    # Save a final run summary
    summary = {
        'best_val_dice': float(best_val_dice),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    }
    with open(exp_dir / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()