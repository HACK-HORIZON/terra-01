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

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from model.hybrid_transformer import OrbitalHybridNet
except ImportError:
    from backend.model.hybrid_transformer import OrbitalHybridNet


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


class EdgeLoss(nn.Module):
    """Laplacian high-frequency edge loss to eliminate blur and maximize sharpness."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        pred_edges = F.conv2d(pred, self.kernel, padding=1, groups=3)
        target_edges = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(pred_edges, target_edges)


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Training] Using device: {device.upper()}")

    model = OrbitalHybridNet(scale=args.scale).to(device)

    # Resolve resume weights (continue training / fine-tune)
    resume_path = args.resume
    if resume_path is None:
        default_weights = os.path.join(args.save_dir, "orbital_hybrid_net.pth")
        if os.path.exists(default_weights) and not args.fresh:
            resume_path = default_weights

    if resume_path and os.path.exists(resume_path) and not args.fresh:
        print(f"[Training] Loading checkpoint to continue training: {resume_path}")
        try:
            state_dict = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("[Training] Checkpoint successfully loaded! Warm-starting training.")
        except Exception as e:
            print(f"[Training] Warning: Could not load checkpoint ({e}). Training from scratch.")
    else:
        print("[Training] Training model from scratch (random initialization).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion_charb = CharbonnierLoss()
    criterion_edge = EdgeLoss().to(device)

    # Auto-detect data_dir if default does not exist
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        alt_dirs = ["./Frontend/dataset", "./dataset", "../Frontend/dataset"]
        for alt in alt_dirs:
            if os.path.exists(alt) and len(os.listdir(alt)) > 0:
                data_dir = alt
                print(f"[Training] Using detected dataset directory: {data_dir}")
                break

    if not os.path.exists(data_dir):
        print(f"[Training] Error: Data directory '{args.data_dir}' not found. Run 'python Frontend/prepare_dataset.py' first.")
        return

    dataset = OrbitalDataset(data_dir, patch_size=args.patch_size, scale=args.scale)
    if len(dataset) == 0:
        print(f"[Training] Error: No valid images found in {data_dir}.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[Training] Dataset loaded: {len(dataset)} orbital tiles. Starting {args.epochs} epochs with Edge-Aware Loss...")

    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = args.save_name if args.save_name else "orbital_hybrid_net.pth"
    save_path = os.path.join(args.save_dir, save_filename)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_mse = 0.0
        start = time.time()
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss_charb = criterion_charb(sr, hr)
            loss_edge = criterion_edge(sr, hr)
            loss = loss_charb + 3.0 * loss_edge
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                mse = F.mse_loss(sr.clamp(0, 1), hr.clamp(0, 1)).item()
                total_mse += mse

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_psnr = 10 * torch.log10(torch.tensor(1.0 / max(avg_mse, 1e-8))).item()
        elapsed = time.time() - start
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | LR: {current_lr:.2e} | Time: {elapsed:.2f}s", flush=True)

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            # Backup previous weights before overwriting
            if os.path.exists(save_path):
                backup_path = save_path + ".bak"
                try:
                    import shutil
                    shutil.copyfile(save_path, backup_path)
                except Exception:
                    pass
            torch.save(model.state_dict(), save_path)
            print(f"[Training] Model checkpoint saved to {save_path} (backup created)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OrbitalHybridNet")
    parser.add_argument("--data_dir", type=str, default="./Frontend/dataset", help="Directory containing HR satellite images")
    parser.add_argument("--save_dir", type=str, default="./backend/model/weights", help="Directory to save model weights")
    parser.add_argument("--save_name", type=str, default="orbital_hybrid_net.pth", help="Checkpoint filename to save")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to continue training from")
    parser.add_argument("--fresh", action="store_true", help="Force train from scratch instead of warm-starting from existing weights")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor")
    parser.add_argument("--patch_size", type=int, default=128, help="Low-resolution input patch size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (use 1e-4 for fine-tuning)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")
    args = parser.parse_args()
    train(args)
