def suggest_hyperparameters(trial):
    """
    Optuna search space for SegFormer (Vision Transformer).
    FAIR rules: Transformers use lower lr than CNNs (more stable).
    - lr: [1e-5, 3e-4] log (lower than CNNs for Transformer stability)
    - weight_decay: [1e-6, 1e-2] log
    - optimizer: AdamW, Adam (Transformers prefer AdamW)
    - backbone: b0, b1, b2 (3 model sizes, not CNN encoders)
    """
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", 
        ["AdamW", "Adam"]
    )
    backbone = trial.suggest_categorical(
        "encoder_name",  # Keep key name for model.py compatibility
        ["nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b2"]
    )
    
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "encoder_name": backbone  # Maps to SegFormer backbone in model.py
    }
