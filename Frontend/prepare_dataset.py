#!/usr/bin/env python
"""
Prepare training dataset from sample images with augmentation.
Creates augmented variations for training the OrbitalHybridNet.
"""
import os
import sys
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'public', 'samples')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
TARGET_SIZE = 512  # HR patch size for training

# Augmentation transforms
augment_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=90, expand=False, fill=0),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
])

def prepare_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
    if not sample_files:
        print("No sample images found!")
        return
    
    print(f"Found {len(sample_files)} base samples")
    
    total_saved = 0
    
    for sample_file in sample_files:
        sample_path = os.path.join(SAMPLES_DIR, sample_file)
        base_name = os.path.splitext(sample_file)[0]
        
        # Load original
        img = Image.open(sample_path).convert("RGB")
        w, h = img.size
        
        # Resize if too small
        min_dim = min(w, h)
        if min_dim < TARGET_SIZE:
            scale = TARGET_SIZE / min_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = img.size
        
        # Generate augmented versions
        # Original + 15 augmented versions per sample
        for i in range(16):
            if i == 0:
                aug_img = img
            else:
                aug_img = augment_transforms(img)
            
            # Random crop to TARGET_SIZE x TARGET_SIZE
            if aug_img.width > TARGET_SIZE or aug_img.height > TARGET_SIZE:
                left = random.randint(0, max(0, aug_img.width - TARGET_SIZE))
                top = random.randint(0, max(0, aug_img.height - TARGET_SIZE))
                right = min(left + TARGET_SIZE, aug_img.width)
                bottom = min(top + TARGET_SIZE, aug_img.height)
                aug_img = aug_img.crop((left, top, right, bottom))
            
            # Resize to exact target if needed
            if aug_img.size != (TARGET_SIZE, TARGET_SIZE):
                aug_img = aug_img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            
            # Save
            out_name = f"{base_name}_aug{i:02d}.png"
            out_path = os.path.join(DATASET_DIR, out_name)
            aug_img.save(out_path, "PNG")
            total_saved += 1
    
    print(f"Created {total_saved} training images in {DATASET_DIR}")

if __name__ == "__main__":
    prepare_dataset()