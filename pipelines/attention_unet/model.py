# Attention U-Net model definition
import json
from pathlib import Path
from monai.networks.nets import AttentionUnet

def get_model(config_override=None):
    if config_override is not None:
        config = config_override
    else:
        config_path = Path("pipelines/attention_unet/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {
                "lr": 3e-4, "weight_decay": 1e-5, "optimizer_name": "AdamW", "base_channels": 16
            }

    base_ch = config.get("base_channels", 16)
    
    # Generate spatial channel configurations mathematically based on base depth
    channel_layout = (base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16)
    
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=channel_layout,
        strides=(2, 2, 2, 2)
    )
    print(f"✅ Medical Attention U-Net instantiated successfully with layout: {channel_layout}")
    return model, config