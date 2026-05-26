import argparse
import os
import torch
import json
import optuna
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.dataset import CarotidPlaqueDataset
from src.losses import HybridLoss
from src.metrics import evaluate_batch
from src.data_augmentation import get_transforms
from src.repro import set_seed, seed_worker

FIXED_SEED = 123
FIXED_SPLIT_PATH = Path("dataset/splits/split_seed123.json")


def load_fixed_split():
    if not FIXED_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Expected split file not found: {FIXED_SPLIT_PATH}"
        )

    with open(FIXED_SPLIT_PATH, "r") as f:
        saved = json.load(f)

    return saved.get("train", []), saved.get("val", [])

def get_model_function(model_name):
    if model_name == "unet":
        from pipelines.unet.model import get_model
    elif model_name == "unet_plus_plus":
        from pipelines.unet_plus_plus.model import get_model
    elif model_name == "attention_unet":
        from pipelines.attention_unet.model import get_model
    elif model_name == "deeplabv3_plus":
        from pipelines.deeplabv3_plus.model import get_model
    elif model_name == "hrnet":
        from pipelines.hrnet.model import get_model
    elif model_name == "segformer":
        from pipelines.segformer.model import get_model
    elif model_name == "my_network":
        from pipelines.my_network.model import get_model
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return get_model

def get_optimizer(optimizer_name, params, lr, weight_decay):
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_hyperparameter_suggester(model_name):
    if model_name == "unet":
        from pipelines.unet.tune import suggest_hyperparameters
    elif model_name == "unet_plus_plus":
        from pipelines.unet_plus_plus.tune import suggest_hyperparameters
    elif model_name == "attention_unet":
        from pipelines.attention_unet.tune import suggest_hyperparameters
    elif model_name == "deeplabv3_plus":
        from pipelines.deeplabv3_plus.tune import suggest_hyperparameters
    elif model_name == "hrnet":
        from pipelines.hrnet.tune import suggest_hyperparameters
    elif model_name == "segformer":
        from pipelines.segformer.tune import suggest_hyperparameters
    elif model_name == "my_network":
        from pipelines.my_network.tune import suggest_hyperparameters
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return suggest_hyperparameters

def objective(trial, args, train_loader, val_loader, device):
    # Clear cache before starting a trial
    torch.cuda.empty_cache()
    
    suggester = get_hyperparameter_suggester(args.model)
    config = suggester(trial)
    
    # Get model with the proposed config
    model_fn = get_model_function(args.model)
    model, _ = model_fn(config_override=config)
    model = model.to(device)
    
    criterion = HybridLoss()
    
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    optimizer_name = config.get("optimizer_name", "AdamW")
    
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    scaler = torch.cuda.amp.GradScaler() # mixed precision scaler
    
    best_val_dice = 0.0
    
    epochs = args.epochs
    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        model.train()
        
        train_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation
        val_loader.dataset.set_epoch(epoch)
        model.eval()
        all_per_image = []
        
        val_bar = tqdm(val_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                per_image, _ = evaluate_batch(outputs, masks)
                all_per_image.extend(per_image)
                
        # Calculate metric
        dice_vals = [m['dice'] for m in all_per_image if m['has_plaque'] and not np.isnan(m['dice'])]
        mean_dice = float(np.nanmean(dice_vals)) if dice_vals else 0.0
        
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            
        trial.report(mean_dice, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_dice

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["unet", "unet_plus_plus", "attention_unet", "deeplabv3_plus", "hrnet", "segformer", "my_network"],
                        help="Specify which architecture to tune.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per trial.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help="Seed. Fixed to 123 for fair comparison across models.")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔧 HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    print(f"📦 Model: {args.model}")
    print(f"🔁 Trials: {args.n_trials}")
    print(f"📚 Epochs per trial: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🌱 Seed: {FIXED_SEED} (fixed for fair comparison)\n")

    if args.seed != FIXED_SEED:
        print(f"⚠️  Overriding --seed {args.seed} with fixed seed {FIXED_SEED} for fair comparison.\n")

    set_seed(FIXED_SEED)

    print("📋 Loading fixed dataset split...")
    train_files, val_files = load_fixed_split()
    print(f"   ✓ Train samples: {len(train_files)}")
    print(f"   ✓ Val samples: {len(val_files)}\n")
    
    print("🖼️  Loading data augmentations...")
    train_transform, val_transform = get_transforms(input_size=512)
    print(f"   ✓ Augmentations loaded\n")
    
    train_dataset = CarotidPlaqueDataset(
        data_dir="dataset/train",
        transform=train_transform,
        seed=FIXED_SEED,
        filenames=train_files,
    )
    val_dataset = CarotidPlaqueDataset(
        data_dir="dataset/train", 
        transform=val_transform,
        seed=FIXED_SEED,
        filenames=val_files,
    )

    g = torch.Generator()
    g.manual_seed(FIXED_SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker
    )

    print("⚡ Creating Optuna study...")
    study = optuna.create_study(direction="maximize", study_name=f"{args.model}_tuning_seed{FIXED_SEED}")
    print(f"   ✓ Study name: {study.study_name}\n")
    
    print("🔍 Starting hyperparameter optimization...")
    print("-" * 70)
    
    def wrapped_objective(trial):
        return objective(trial, args, train_loader, val_loader, device)

    study.optimize(wrapped_objective, n_trials=args.n_trials)

    print("-" * 70)
    print("\n✨ TUNING COMPLETED!\n")
    
    print("🏆 Best Trial Results:")
    print("-" * 70)
    print(f"Trial Number: {study.best_trial.number}")
    print(f"Best Dice Score: {study.best_trial.value:.6f}\n")
    
    print("Optimized Hyperparameters:")
    print("-" * 70)
    for key, value in study.best_trial.params.items():
        print(f"  {key:.<30} {value}")
    print("-" * 70 + "\n")

    # Save to config file
    config_path = Path(f"pipelines/{args.model}/config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    print(f"💾 Configuration saved: {config_path}\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()