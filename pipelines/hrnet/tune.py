def suggest_hyperparameters(trial):
    """
    Optuna search space for HRNet.
    FAIR rules: NO encoder search (HRNet is a complete architecture).
    - lr: [1e-5, 3e-3] log (higher range than Transformers due to CNN nature)
    - weight_decay: [1e-6, 1e-2] log
    - optimizer: AdamW, Adam, SGD (standard optimizers)
    Note: HRNet backbone variants (w32, w48) are fixed, not tuned.
    """
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", 
        ["AdamW", "Adam", "SGD"]
    )
    
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
    }
