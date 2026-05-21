def suggest_hyperparameters(trial):
    """
    Optuna search space for nnU-Net.
    FAIR rules: nnU-Net must NOT be heavily tuned (already optimized out-of-the-box).
    - Optional tuning: lr [1e-5, 1e-3] log only
    - Use nnU-Net defaults for all other hyperparameters
    
    nnU-Net is designed to be a baseline and automatic method, so excessive tuning
    would be unfair to its original design philosophy. Minimal tuning ensures
    benchmark fairness by letting nnU-Net work as intended.
    """
    # Optional: tune only learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    return {
        "lr": lr,
        # All other nnU-Net hyperparameters use pipeline defaults
    }
