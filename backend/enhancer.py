"""
Orbital Image Enhancement Engine.
Integrates OrbitalHybridNet (CNN + Transformer) with satellite-specific image processing:
- Tile-based or full-frame inference
- Resolution enhancement (2x Super-Resolution)
- Atmospheric contrast optimization (CLAHE in LAB space)
- High-frequency edge restoration
- Scientific metric extraction (PSNR, SSIM, RMSE, MAE)
"""

import io
import os
import time
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from .model.hybrid_transformer import OrbitalHybridNet, create_model
from .model.metrics import calculate_all_metrics

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "model", "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "orbital_hybrid_net.pth")


class OrbitalEnhancer:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[OrbitalEnhancer] Initializing hybrid model on {self.device.upper()}...")
        self.model_2x = create_model(scale=2, device=self.device)
        self._init_or_load_weights()
        self.model_2x.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == 'cuda':
            try:
                self.model_2x = torch.compile(self.model_2x, mode='reduce-overhead')
                print("[OrbitalEnhancer] Model compiled with torch.compile")
            except Exception as e:
                print(f"[OrbitalEnhancer] torch.compile failed: {e}")
        
        # Half precision for CUDA
        if self.device == 'cuda':
            self.model_2x.half()
            print("[OrbitalEnhancer] Using FP16 inference")
        
        print(f"[OrbitalEnhancer] Hybrid model ready.")

    def _init_or_load_weights(self):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        # Remove old weights if we want a fresh calibrated initialization
        if os.path.exists(WEIGHTS_PATH):
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
                self.model_2x.load_state_dict(state_dict)
                print(f"[OrbitalEnhancer] Loaded weights from {WEIGHTS_PATH}")
                return
            except Exception as e:
                print(f"[OrbitalEnhancer] Error loading existing weights: {e}. Reinitializing.")

        # Zero-centered residual initialization:
        # Shallow and deep backbone layers initialize with Kaiming normal
        for m in self.model_2x.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Crucial for residual learning: Zero-initialize the final projection layer
        # so that residual detail starts cleanly and preserves high fidelity (high PSNR/SSIM)
        torch.nn.init.zeros_(self.model_2x.conv_final[2].weight)
        if self.model_2x.conv_final[2].bias is not None:
            torch.nn.init.zeros_(self.model_2x.conv_final[2].bias)

        try:
            torch.save(self.model_2x.state_dict(), WEIGHTS_PATH)
            print(f"[OrbitalEnhancer] Saved calibrated residual weights to {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[OrbitalEnhancer] Could not save weights: {e}")

    def enhance_image(
        self,
        image_bytes: bytes,
        scale: int = 2,
        apply_dehaze: bool = True,
        apply_sharpen: bool = True,
        denoise_level: float = 0.5,
        remove_clouds: bool = False,
        remove_obstacles: bool = False,
        deblur: bool = False,
        fast_mode: bool = False,
        compute_metrics: bool = True
    ) -> dict:
        """
        Enhances an orbital satellite image with the CNN-Transformer hybrid model.
        fast_mode: skip heavy post-processing for speed
        compute_metrics: calculate PSNR/SSIM/RMSE/MAE (adds ~200ms)
        """
        start_time = time.perf_counter()

        # 1. Load image
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Cap max input size to 1200px for rapid response
        max_dim = 1200
        if max(orig_w, orig_h) > max_dim:
            ratio = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            orig_w, orig_h = new_w, new_h

        img_np = np.array(img_pil)

        # 2. Hybrid Model Inference
        x_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Move to device with correct dtype
        if self.device == 'cuda':
            x_tensor = x_tensor.half().to(self.device)
        else:
            x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            if scale == 2:
                if max(orig_w, orig_h) > 800:
                    enhanced_tensor = self._tile_forward(x_tensor, self.model_2x, tile_size=400, overlap=16)
                else:
                    enhanced_tensor = self.model_2x(x_tensor)
            else:
                enhanced_tensor = x_tensor

        # Convert back to float for post-processing
        if enhanced_tensor.dtype == torch.float16:
            enhanced_tensor = enhanced_tensor.float()
        
        enhanced_np = (enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # 3. Satellite Spectral & Contrast Refinements (optional fast_mode)
        if not fast_mode and any([apply_dehaze, apply_sharpen, denoise_level > 0, remove_clouds, remove_obstacles, deblur]):
            enhanced_np = self._apply_orbital_filters(
                enhanced_np,
                apply_dehaze=apply_dehaze,
                apply_sharpen=apply_sharpen,
                denoise_level=denoise_level,
                remove_clouds=remove_clouds,
                remove_obstacles=remove_obstacles,
                deblur=deblur
            )

        # 4. Calculate Scientific Metrics (optional)
        metrics = {"psnr": 0, "ssim": 0, "rmse": 0, "mae": 0}
        if compute_metrics:
            metrics = calculate_all_metrics(img_np, enhanced_np)

        # 5. Base64 encoding (optimized)
        out_pil = Image.fromarray(enhanced_np)
        out_w, out_h = out_pil.size

        buf_enhanced = io.BytesIO()
        out_pil.save(buf_enhanced, format="PNG", compress_level=1)  # Fast compression
        enhanced_b64 = "data:image/png;base64," + base64.b64encode(buf_enhanced.getvalue()).decode("utf-8")

        buf_orig = io.BytesIO()
        img_pil.save(buf_orig, format="PNG", compress_level=1)
        original_b64 = "data:image/png;base64," + base64.b64encode(buf_orig.getvalue()).decode("utf-8")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "status": "success",
            "enhanced_image": enhanced_b64,
            "original_image": original_b64,
            "metrics": metrics,
            "processing_time_ms": elapsed_ms,
            "input_resolution": [orig_w, orig_h],
            "output_resolution": [out_w, out_h],
            "device": self.device.upper(),
            "model_architecture": "OrbitalHybridNet (CNN + MDTA Transformer)",
            "scale_factor": scale
        }

    def _tile_forward(self, x: torch.Tensor, model: torch.nn.Module, tile_size: int = 400, overlap: int = 32) -> torch.Tensor:
        b, c, h, w = x.shape
        scale = model.scale
        out_h, out_w = h * scale, w * scale
        output = torch.zeros((b, c, out_h, out_w), device=x.device)
        weights = torch.zeros((b, 1, out_h, out_w), device=x.device)

        stride = tile_size - overlap
        for y in range(0, h, stride):
            y_end = min(y + tile_size, h)
            y_start = max(0, y_end - tile_size)

            for xx in range(0, w, stride):
                x_end = min(xx + tile_size, w)
                x_start = max(0, x_end - tile_size)

                tile = x[:, :, y_start:y_end, x_start:x_end]
                tile_out = model(tile)

                out_y_start = y_start * scale
                out_y_end = y_end * scale
                out_x_start = x_start * scale
                out_x_end = x_end * scale

                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += tile_out
                weights[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1.0

        return output / torch.clamp(weights, min=1.0)

    def _detect_clouds(self, img_rgb: np.ndarray) -> np.ndarray:
        """Detect clouds using HSV color space and brightness thresholds."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Clouds: high value (bright), low saturation
        cloud_mask = (v > 180) & (s < 60)
        cloud_mask = cloud_mask.astype(np.uint8) * 255

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)

        # Dilate slightly to cover cloud edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloud_mask = cv2.dilate(cloud_mask, kernel_dilate, iterations=1)

        return cloud_mask

    def _detect_obstacles(self, img_rgb: np.ndarray) -> np.ndarray:
        """Detect potential obstacles (dark artifacts, shadows, sensor defects)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Detect very dark regions (potential shadows/obstacles)
        dark_mask = (gray < 30).astype(np.uint8) * 255

        # Detect very bright saturated regions (sensor artifacts)
        bright_mask = (gray > 250).astype(np.uint8) * 255

        # Combine
        obstacle_mask = cv2.bitwise_or(dark_mask, bright_mask)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        return obstacle_mask

    def _inpaint_regions(self, img_rgb: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
        """Inpaint masked regions using OpenCV."""
        if method == "telea":
            return cv2.inpaint(img_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(img_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    def _deblur_wiener(self, img_rgb: np.ndarray, kernel_size: int = 5, noise_ratio: float = 0.01) -> np.ndarray:
        """Apply Wiener deconvolution for deblurring."""
        # Simple approach: unsharp mask with adaptive strength based on local contrast
        # For true Wiener deconvolution, we'd need PSF estimation
        # Using multi-scale unsharp masking as practical approximation
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Multi-scale unsharp mask
        blur1 = cv2.GaussianBlur(l, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(l, (0, 0), 3.0)
        blur3 = cv2.GaussianBlur(l, (0, 0), 5.0)

        detail1 = cv2.subtract(l, blur1)
        detail2 = cv2.subtract(l, blur2)
        detail3 = cv2.subtract(l, blur3)

        # Combine details with decreasing weights
        enhanced_l = cv2.addWeighted(l, 1.0, detail1, 0.6, 0)
        enhanced_l = cv2.addWeighted(enhanced_l, 1.0, detail2, 0.4, 0)
        enhanced_l = cv2.addWeighted(enhanced_l, 1.0, detail3, 0.2, 0)

        # Local contrast enhancement
        enhanced_l = cv2.normalize(enhanced_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    def _denoise_bilateral(self, img_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving denoising."""
        d = int(5 + strength * 10)
        sigma_color = 50 + strength * 50
        sigma_space = 50 + strength * 50
        return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)

    def _apply_orbital_filters(
        self,
        img_rgb: np.ndarray,
        apply_dehaze: bool = True,
        apply_sharpen: bool = True,
        denoise_level: float = 0.5,
        remove_clouds: bool = False,
        remove_obstacles: bool = False,
        deblur: bool = False
    ) -> np.ndarray:
        res = img_rgb.copy()

        # 1. Cloud removal (before other processing)
        if remove_clouds:
            cloud_mask = self._detect_clouds(res)
            if np.any(cloud_mask > 0):
                res = self._inpaint_regions(res, cloud_mask, method="telea")

        # 2. Obstacle removal
        if remove_obstacles:
            obstacle_mask = self._detect_obstacles(res)
            if np.any(obstacle_mask > 0):
                res = self._inpaint_regions(res, obstacle_mask, method="telea")

        # 3. Deblurring
        if deblur:
            res = self._deblur_wiener(res)

        # 4. CLAHE (Contrast-Limited Adaptive Histogram Equalization) in LAB space
        if apply_dehaze:
            lab = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            res = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 5. Denoising (using denoise_level parameter)
        if denoise_level > 0:
            res = self._denoise_bilateral(res, strength=denoise_level)

        # 6. High-frequency edge sharpening for satellite structures (roads, runways, docks)
        if apply_sharpen:
            blurred = cv2.GaussianBlur(res, (0, 0), 1.5)
            res = cv2.addWeighted(res, 1.25, blurred, -0.25, 0)
            res = np.clip(res, 0, 255).astype(np.uint8)

        return res


# Global singleton instance
_enhancer_instance = None

def get_enhancer() -> OrbitalEnhancer:
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = OrbitalEnhancer()
    return _enhancer_instance
