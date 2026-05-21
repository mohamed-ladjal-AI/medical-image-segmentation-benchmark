# SegFormer configuration
import json
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

def get_model(config_override=None):
    if config_override is not None:
        config = config_override
    else:
        config_path = Path("pipelines/segformer/config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {
                "lr": 5e-5, "weight_decay": 1e-4, "optimizer_name": "AdamW", "encoder_name": "nvidia/mit-b0"
            }

    chosen_backbone = config.get("encoder_name", "nvidia/mit-b0")
    
    # Sourced from official Transformer Hub
    model = SegformerForSemanticSegmentation.from_pretrained(
        chosen_backbone,
        num_labels=1,
        num_channels=1, # Overrides 3-channel RGB default to 1-channel grayscale safely
        ignore_mismatched_sizes=True
    )
    
    # Wrap model forward pass to output only the logits tensor 
    # to match expectations in run_benchmark.py loop
    class SegformerWrapper(model.__class__):
        def forward(self, x):
            outputs = super().forward(pixel_values=x)
            # Upsample logits to match original 512x512 resolution
            # Return raw logits (sigmoid applied in loss only for AMP compatibility)
            import torch.nn.functional as F
            return F.interpolate(outputs.logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    model.__class__ = SegformerWrapper
    print(f"✅ Vision Transformer SegFormer loaded successfully: {chosen_backbone}")
    return model, config