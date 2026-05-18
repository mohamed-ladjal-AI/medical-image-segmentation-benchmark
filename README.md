# Carotid Plaque Segmentation Benchmark
Reproducible Deep Learning Framework.

## Reproducibility Checklist

- Fixed global seed via `--seed` and deterministic PyTorch settings.
- Per-sample augmentation seed derived from `seed + epoch + sample_index`, so Albumentations randomness is controlled at the item level rather than left to worker order.
- DataLoader workers are seeded explicitly, and the training set exposes epoch state so augmentation seeds remain stable across runs.
- Each experiment writes `run_args.json`, `model_config.json`, `run_summary.json`, TensorBoard logs, and checkpoint weights into a dedicated output directory.

## Usage

### Training with Tuned Hyperparameters

Each model includes a `config.json` with tuned hyperparameters (learning rate, weight decay, encoder choice, etc.). Use the `--hyperparameter` flag to load them:

```bash
python run_benchmark.py --model unet --epochs 50 --batch_size 8 --seed 123 --hyperparameter pipelines/unet/config.json --exp_name baseline
```

If `--hyperparameter` is omitted, the training will use default hyperparameters from `get_model()`.

### Hyperparameter Tuning

Each model's `tune.py` script runs Optuna to find optimal hyperparameters. After tuning completes, update the respective `config.json`:

```bash
cd pipelines/unet
python tune.py  # Outputs optimal hyperparameters
# Copy the best trial's config to config.json
```

Each `config.json` should contain:
```json
{
  "lr": 3e-4,
  "weight_decay": 1e-5,
  "optimizer_name": "AdamW",
  "encoder_name": "resnet34"
}
```

### Example Workflow

```bash
# 1. Tune hyperparameters (one-time per model)
python pipelines/unet/tune.py

# 2. Save tuned config to pipelines/unet/config.json

# 3. Train with frozen hyperparameters (reproducible runs)
python run_benchmark.py --model unet --epochs 100 --batch_size 8 --seed 42 --hyperparameter pipelines/unet/config.json --exp_name trial1
python run_benchmark.py --model unet --epochs 100 --batch_size 8 --seed 123 --hyperparameter pipelines/unet/config.json --exp_name trial2
```

All experiment artifacts are saved to `experiments/<model>_<exp_name>_seed<seed>_<timestamp>/`.
