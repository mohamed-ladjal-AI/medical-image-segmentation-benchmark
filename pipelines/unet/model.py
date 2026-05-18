# Standard U-Net model definition
import json
from pathlib import Path
import segmentation_models_pytorch as smp

def get_model(config_override=None):
    """
    Factory hook for U-Net.
    1. If config_override is passed, it uses it (for Optuna trials).
    2. If not, it looks for the saved JSON file from a past tuning run.
    3. If no JSON exists, it falls back to a safe baseline default.
    """
    # 1. Determine which configuration to use
    if config_override is not None:
        config = config_override
    else:
        # Look for the saved Optuna configuration card
        config_path = Path("pipelines/unet/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            print("💡 Connected: Loaded optimized parameters from Optuna profile.")
        else:
            print("⚠️ No Optuna profile found. Utilizing baseline defaults.")
            config = {
                "lr": 3e-4, 
                "weight_decay": 1e-5, 
                "optimizer_name": "AdamW",
                "encoder_name": "resnet34"
            }

    # 2. Extract the encoder chosen by you or Optuna (e.g., 'resnet34', 'resnet50')
    chosen_encoder = config.get("encoder_name", "resnet34")
    
    # 3. Instantiate the verified U-Net from the SMP library
    model = smp.Unet(
        encoder_name=chosen_encoder,     # Dynamic backbone selection
        encoder_weights="imagenet",      # Download pre-trained ImageNet weights automatically
        in_channels=1,                   # 1 channel because your ultrasound images are grayscale
        classes=1,                       # 1 class output for binary plaque mask
        activation="sigmoid"             # Squashes pixel outputs between 0.0 and 1.0
    )

    print(f"✅ Pretrained encoder loaded successfully: {chosen_encoder} (encoder_weights=imagenet)")
    
    return model, config