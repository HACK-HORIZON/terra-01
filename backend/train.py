"""
OrbitalHybridNet Training & Fine-Tuning Script.

Allows training or fine-tuning the CNN-Transformer hybrid model on custom
satellite/orbital image datasets (Sentinel-2, Landsat, SpaceNet, UC Merced).

Usage:
    python backend/train.py --data_dir ./dataset/satellite_images --epochs 25 --batch_size 8 --lr 2e-4
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from model.hybrid_transformer import OrbitalHybridNet


class OrbitalDataset(Dataset):
    """Dataset for training super-resolution on high-resolution satellite imagery."""
    def __init__(self, folder: str, patch_size: int = 128, scale: int = 2):
        self.folder = folder
        self.patch_size = patch_size
        self.scale = scale
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        self.crop = T.RandomCrop(patch_size * scale)
        self.flip = T.RandomHorizontalFlip()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        hr_img = Image.open(path).convert("RGB")
        # Ensure image is large enough
        target_size = self.patch_size * self.scale
        if hr_img.width < target_size or hr_img.height < target_size:
            hr_img = hr_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        hr_tensor = T.ToTensor()(self.crop(hr_img))
        # Downsample to create LR input
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0),
            size=(self.patch_size, self.patch_size),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        return lr_tensor, hr_tensor


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1) widely used in image restoration."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Training] Using device: {device}")

    model = OrbitalHybridNet(scale=args.scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CharbonnierLoss()

    if not os.path.exists(args.data_dir):
        print(f"[Training] Data directory {args.data_dir} does not exist. Please provide dataset path.")
        return

    dataset = OrbitalDataset(args.data_dir, patch_size=args.patch_size, scale=args.scale)
    if len(dataset) == 0:
        print(f"[Training] No images found in {args.data_dir}.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[Training] Dataset loaded: {len(dataset)} orbital tiles. Starting {args.epochs} epochs...")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"orbital_hybrid_net_s{args.scale}.pth")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        start = time.time()
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Avg Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), save_path)
            print(f"[Training] Model checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OrbitalHybridNet")
    parser.add_argument("--data_dir", type=str, default="./dataset/satellite_images")
    parser.add_argument("--save_dir", type=str, default="./backend/model/weights")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)
