import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from src.repro import derive_sample_seed, temporary_seed

class CarotidPlaqueDataset(Dataset):
    def __init__(self, data_dir, transform=None, seed=42, filenames=None):
        """
        Args:
            data_dir (str): Path to the folder containing both images and masks 
                            (e.g., 'dataset/train' or 'dataset/val')
            transform (albumentations.Compose): Optional augmentations pipeline.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.seed = seed
        self.epoch = 0
        # If a specific filenames list is provided, use it (useful for train/val splits)
        if filenames is not None:
            self.image_filenames = list(filenames)
        else:
            # 1. Grab all files in the directory
            all_files = os.listdir(data_dir)

            # 2. Extract ONLY the raw images (files that DO NOT have '_labeled' in their name)
            # Supported extensions: .png, .jpg, .jpeg
            self.image_filenames = sorted([
                f for f in all_files 
                if "_labeled" not in f and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

        print(f"👁️ Found {len(self.image_filenames)} valid raw ultrasound frames in: {data_dir}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Determine paths
        img_name = self.image_filenames[idx]
        
        # Split extension out (e.g., '1.png' -> base='1', ext='.png')
        base_name, ext = os.path.splitext(img_name)
        mask_name = f"{base_name}_labeled{ext}"
        
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        # Verify the mask file actually exists to protect against missing annotations
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing matching mask file: {mask_path} for image {img_path}")

        # Load image and mask as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert mask to explicit binary 0.0 or 1.0 representation
        mask = (mask > 127).astype(np.float32)

        # Apply spatial & pixel-level transformations identically
        if self.transform:
            sample_seed = derive_sample_seed(self.seed, self.epoch, idx)
            with temporary_seed(sample_seed):
                augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Format into PyTorch Tensor structure [Channels, Height, Width]
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor