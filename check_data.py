import matplotlib.pyplot as plt
from src.dataset import CarotidPlaqueDataset

# Define paths pointing to your local data
TRAIN_IMG_DIR = "dataset/train/"

try:
    dataset = CarotidPlaqueDataset(image_dir=TRAIN_IMG_DIR)
    print(f"Successfully loaded {len(dataset)} training pairs.")
except Exception as e:
    print(f"Error checking data layout: {e}\nMake sure 'dataset/train/images' exists and contains files.")
