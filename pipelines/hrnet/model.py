# HRNet model definition
import json
from pathlib import Path
import segmentation_models_pytorch as smp

def get_model(config_override=None):
    if config_override is not None:
        config = config_override
    else:
        config_path = Path("pipelines/hrnet/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            print("💡 Connected: Loaded optimized parameters from Optuna profile.")
        else:
            print("⚠️ No Optuna profile found. Utilizing baseline defaults.")
            config = {
                "lr": 2e-4, "weight_decay": 1e-5, "optimizer_name": "AdamW", "encoder_name": "hrnet_w32"
            }

    chosen_encoder = config.get("encoder_name", "hrnet_w32") # Optuna handles hrnet_w32 or hrnet_w48
    
    model = smp.Unet(
        encoder_name=chosen_encoder,
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation="sigmoid"
    )
    print(f"✅ Pretrained High-Res Backbone loaded successfully: {chosen_encoder}")
    return model, config