def suggest_hyperparameters(trial):
    """
    Suggests hyperparameters for the hrnet architecture.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "AdamW", "SGD"])
    
    # Specific hyperparameters for hrnet
    encoder_name = trial.suggest_categorical("encoder_name", ["resnet18", "resnet34", "resnet50"])
    
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "encoder_name": encoder_name
    }
