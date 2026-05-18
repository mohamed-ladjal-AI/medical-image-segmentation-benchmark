#!/usr/bin/env python3
"""Create a deterministic train/val split from dataset/train and save to a JSON file.

Usage:
  python scripts/make_split.py --seed 42 --split_ratio 0.1 --out dataset/splits/split_seed42.json

If --out is omitted the default is dataset/splits/split_seed<seed>.json
"""
import argparse
import os
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Create deterministic train/val split from dataset/train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_ratio", type=float, default=0.1)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    train_dir = Path("dataset/train")
    if not train_dir.exists():
        raise FileNotFoundError("dataset/train not found. Place training images and masks in dataset/train")

    files = sorted([f for f in os.listdir(train_dir) if "_labeled" not in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(files) == 0:
        raise FileNotFoundError("No training images found in dataset/train")

    if not (0.0 < args.split_ratio < 0.5):
        raise ValueError("--split_ratio must be between 0 and 0.5")

    out_path = Path(args.out) if args.out else Path("dataset/splits") / f"split_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import random
    rnd = random.Random(args.seed)
    files_shuffled = list(files)
    rnd.shuffle(files_shuffled)

    val_count = max(1, int(len(files_shuffled) * args.split_ratio))
    val_files = files_shuffled[:val_count]
    train_files = files_shuffled[val_count:]

    with open(out_path, 'w') as f:
        json.dump({'train': train_files, 'val': val_files}, f, indent=2)

    print(f"Saved split to {out_path}")
    print(f"Total images: {len(files)}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} ({len(val_files)/len(files):.1%})")
    print("Example validation files:")
    for fname in val_files[:10]:
        print(f"  {fname}")


if __name__ == '__main__':
    main()
