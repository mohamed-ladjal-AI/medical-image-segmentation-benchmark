def suggest_hyperparameters(trial):
    """
    Optuna search space for Attention U-Net (MONAI).
    MONAI AttentionUnet builds its own channel-based architecture, not
    an encoder-decoder — no external encoder_name to tune.
    - lr: [1e-5, 3e-3] log (standard CNN learning rate range)
    - weight_decay: [1e-6, 1e-2] log
    - optimizer: AdamW, Adam, SGD (standard optimizers)
    - base_channels: [8, 16, 32] (controls model capacity: 8=slim, 32=wide)
    """
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer_name",
        ["AdamW", "Adam", "SGD"]
    )
    base_channels = trial.suggest_categorical(
        "base_channels",
        [8, 16, 32]
    )

    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "base_channels": base_channels,
    }
