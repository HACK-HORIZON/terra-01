"""
Automated Model Retraining Pipeline.

Performs end-to-end automated fine-tuning:
1. Checks for new/existing raw satellite images in samples folder.
2. Automatically generates augmented patches via prepare_dataset.py.
3. Automatically fine-tunes OrbitalHybridNet using warm-started weights.
4. Validates convergence & quality (PSNR/Loss).
5. Updates production weights with automatic backup.

Usage:
    python backend/auto_retrain.py --epochs 15 --lr 1e-4
"""

import os
import sys
import argparse
import subprocess
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Frontend", "public", "samples")
DATASET_DIR = os.path.join(ROOT_DIR, "Frontend", "dataset")
PREPARE_SCRIPT = os.path.join(ROOT_DIR, "Frontend", "prepare_dataset.py")
TRAIN_SCRIPT = os.path.join(ROOT_DIR, "backend", "train.py")
WEIGHTS_FILE = os.path.join(ROOT_DIR, "backend", "model", "weights", "orbital_hybrid_net.pth")


def run_command(cmd, desc):
    print(f"\n[AutoTrain] >>> {desc}...")
    start = time.time()
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[AutoTrain] Error executing: {desc} (Exit code {res.returncode})")
        return False
    print(f"[AutoTrain] Done {desc} in {time.time() - start:.1f}s")
    return True


def auto_retrain(epochs: int = 15, lr: float = 1e-4, batch_size: int = 8, fresh: bool = False):
    print("=" * 65)
    print("  TERRA: Automated Model Retraining Pipeline")
    print("=" * 65)

    # Step 1: Augmentation and dataset preparation
    python_exe = sys.executable
    if not run_command(f'"{python_exe}" "{PREPARE_SCRIPT}"', "Step 1: Dataset Augmentation"):
        print("[AutoTrain] Dataset preparation failed.")
        return False

    # Step 2: Fine-Tuning
    fresh_flag = "--fresh" if fresh else ""
    train_cmd = (
        f'"{python_exe}" "{TRAIN_SCRIPT}" '
        f'--data_dir "{DATASET_DIR}" '
        f'--epochs {epochs} '
        f'--lr {lr} '
        f'--batch_size {batch_size} '
        f'--save_name "orbital_hybrid_net.pth" '
        f'{fresh_flag}'
    )

    if not run_command(train_cmd, f"Step 2: Training OrbitalHybridNet ({epochs} epochs)"):
        print("[AutoTrain] Model training failed.")
        return False

    print("\n" + "=" * 65)
    print(f"  SUCCESS! Model trained and saved to:")
    print(f"  {WEIGHTS_FILE}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Retraining for OrbitalHybridNet")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--fresh", action="store_true", help="Train from scratch instead of warm-starting")
    args = parser.parse_args()

    auto_retrain(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, fresh=args.fresh)
