import albumentations as A


def get_transforms(input_size: int = 512,
                   hflip_p: float = 0.5,
                   brightness_p: float = 0.2,
                   affine_p: float = 0.5):
    """Return (train_transform, val_transform) for the benchmark.

    Parameters are exposed so pipeline configs can override them later.
    """
    train_transform = A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=hflip_p),
        A.RandomBrightnessContrast(p=brightness_p),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=15, p=affine_p, mode=0),
    ])

    val_transform = A.Compose([
        A.Resize(input_size, input_size),
    ])

    return train_transform, val_transform
