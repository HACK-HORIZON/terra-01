"""
Scientific Quality Metrics for Orbital and Satellite Imagery Enhancement.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio) in dB [typically 28 to 45 dB]
- SSIM (Structural Similarity Index Measure) [0 to 1, standard 11x11 Gaussian window]
- RMSE (Root Mean Squared Error) [0 to 255 scale]
- MAE (Mean Absolute Error) [0 to 255 scale]
"""

import math
import numpy as np
import cv2
from typing import Dict


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-10:
        return 50.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Standard Structural Similarity Index (SSIM) across channels
    using an 11x11 Gaussian sliding window with sigma=1.5 (Wang et al., 2004).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3:
        ssims = []
        for c in range(img1.shape[2]):
            ssims.append(_ssim_channel_gaussian(img1[:, :, c], img2[:, :, c], max_val))
        return float(np.mean(ssims))
    else:
        return float(_ssim_channel_gaussian(img1, img2, max_val))


def _ssim_channel_gaussian(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ksize = 11
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(img2, (ksize, ksize), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (ksize, ksize), sigma) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / np.maximum(den, 1e-10)
    return float(np.mean(ssim_map))


def calculate_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE) on a 0-255 scale."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return float(np.sqrt(mse))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) on a 0-255 scale."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return float(np.mean(np.abs(img1 - img2)))


def calculate_all_metrics(
    reference: np.ndarray,
    enhanced: np.ndarray,
    max_val: float = 255.0
) -> Dict[str, float]:
    """
    Calculate PSNR, SSIM, RMSE, and MAE.
    If the enhanced image has higher spatial resolution (e.g. 2x super-resolution),
    the reference image is bicubically resized to compute structural and fidelity metrics.
    """
    # For large images (>480px), evaluate metrics on a normalized proxy to keep RAM tiny and compute in <10ms
    max_metric_dim = 480
    if max(w_enh, h_enh) > max_metric_dim:
        scale_factor = max_metric_dim / max(w_enh, h_enh)
        m_w, m_h = int(w_enh * scale_factor), int(h_enh * scale_factor)
        enhanced_sub = cv2.resize(enhanced, (m_w, m_h), interpolation=cv2.INTER_AREA)
        ref_sub = cv2.resize(reference, (m_w, m_h), interpolation=cv2.INTER_CUBIC)
    else:
        enhanced_sub = enhanced
        if (h_ref, w_ref) != (h_enh, w_enh):
            ref_sub = cv2.resize(reference, (w_enh, h_enh), interpolation=cv2.INTER_CUBIC)
        else:
            ref_sub = reference

    psnr = calculate_psnr(ref_sub, enhanced_sub, max_val=max_val)
    ssim = calculate_ssim(ref_sub, enhanced_sub, max_val=max_val)
    rmse = calculate_rmse(ref_sub, enhanced_sub)
    mae = calculate_mae(ref_sub, enhanced_sub)

    return {
        "psnr": round(psnr, 2),
        "ssim": round(min(max(ssim, 0.0), 1.0), 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }
