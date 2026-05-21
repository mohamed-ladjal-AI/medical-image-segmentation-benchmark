def suggest_hyperparameters(trial):
    """
    Optuna search space for DeepLabV3+ (CNN backbone).
    FAIR rules: shared with U-Net, U-Net++, Attention U-Net.
    - lr: [1e-5, 3e-3] log (standard CNN learning rate range)
    - weight_decay: [1e-6, 1e-2] log
    - optimizer: AdamW, Adam, SGD (standard optimizers)
    - encoder: resnet18/34/50, efficientnet-b0/b3 (5 backbones)
    """
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", 
        ["AdamW", "Adam", "SGD"]
    )
    encoder_name = trial.suggest_categorical(
        "encoder_name", 
        ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b3"]
    )
    
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "encoder_name": encoder_name
    }
