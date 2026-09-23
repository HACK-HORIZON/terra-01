# Terra — Orbital Image Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/Vite-8.1-646cff.svg)](https://vitejs.dev)
[![React](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Terra** transforms orbital and satellite imagery quality with intelligent CNN-Transformer hybrid deep learning. Achieve 2× super-resolution with measurable scientific metrics (PSNR, SSIM, RMSE, MAE) — all running in your browser via a beautiful interactive frontend.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm` (or `pnpm`)
- **GPU (optional)**: CUDA 11.8+ for accelerated inference/training

### 1. Clone & Install Dependencies

```bash
# Python dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
npm install
```

### 2. Run the Platform

**Option A: FastAPI only (recommended — single port, production-like)**
```bash
python run.py
# → http://localhost:8000
```

**Option B: FastAPI + Vite Dev Server (for frontend development with HMR)**
```bash
# Terminal 1: FastAPI backend
python run.py

# Terminal 2: Vite dev server (proxies API to FastAPI)
npm run dev
# → http://localhost:8080
```

---

## 🏗️ Project Architecture

```
Terra/
├── backend/                          # FastAPI + PyTorch Backend
│   ├── app.py                        # FastAPI application & routes
│   ├── enhancer.py                   # OrbitalEnhancer inference engine
│   ├── train.py                      # Training/fine-tuning script
│   ├── generate_samples.py           # Sample data generator
│   ├── __init__.py
│   └── model/
│       ├── hybrid_transformer.py     # OrbitalHybridNet architecture
│       ├── metrics.py                # PSNR/SSIM/RMSE/MAE calculations
│       ├── __init__.py
│       └── weights/
│           └── orbital_hybrid_net.pth  # Pre-trained weights (545K params)
│
├── public/                           # Static assets served by FastAPI
│   ├── samples/                      # Demo satellite images (airport, agricultural, coastal)
│   ├── favicon.ico
│   └── robots.txt
│
├── dataset/                          # Generated training data (gitignored)
│   └── *.png                         # Augmented 512×512 patches
│
├── index.html                        # Main SPA entry point
├── script.js                         # Frontend logic (vanilla JS module)
├── styles.css                        # Tailwind + custom styles
├── run.py                            # Application launcher
├── prepare_dataset.py                # Dataset preparation script
├── package.json                      # Node.js dependencies & scripts
├── vite.config.ts                    # Vite config with API proxy
├── tsconfig.json                     # TypeScript config
├── tailwind.config.ts                # Tailwind CSS config
└── postcss.config.js                 # PostCSS config
```

### High-Level Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Browser UI    │────▶│  FastAPI /api    │────▶│  OrbitalEnhancer    │
│  (index.html +  │     │  /enhance        │     │  (CPU/GPU)          │
│   script.js)    │◀────│  (REST + JSON)   │◀────│  OrbitalHybridNet   │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────────┐
                        │ Static Assets    │     │ Post-Processing     │
                        │ /samples, CSS,   │     │ CLAHE, Sharpening,  │
                        │ JS, favicon      │     │ Deblur, Inpainting  │
                        └──────────────────┘     └─────────────────────┘
```

---

## 🧠 Model Architecture: OrbitalHybridNet

### Design Philosophy

OrbitalHybridNet fuses **CNN local feature extraction** with **Transformer global context modeling** — purpose-built for orbital/satellite imagery where:
- **Local textures** (roads, runways, crop rows, dock edges) need precise CNN kernels
- **Global structures** (urban grids, coastlines, cloud systems) benefit from long-range attention
- **Linear complexity** attention enables 512×512+ tile processing without quadratic cost

### Architecture Diagram

```
Input (3, H, W)
      │
      ▼
┌─────────────────────────────────────┐
│  Shallow Feature Extractor (CNN)    │
│  Conv3×3 → LeakyReLU → Conv3×3      │  dim=48
└─────────────────────────────────────┘
      │              │
      │              ▼ (long skip)
      │    ┌─────────────────────────┐
      │    │  Deep Hybrid Backbone   │
      │    │  ×3 Hybrid Groups       │
      │    │  ┌───────────────────┐  │
      │    │  │ ResidualCNNBlock  │  │  ← Local spatial-channel (RCAB)
      │    │  │ TransformerBlock  │  │  ← MDTA (O(HW) attention)
      │    │  │ GatedFeedForward  │  │  ← GFFN (depthwise + GELU)
      │    │  └───────────────────┘  │
      │    └─────────────────────────┘
      │              │
      ▼              ▼
┌─────────────────────────────────────┐
│  ConvAfterBackbone + Shallow Skip   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  PixelShuffle Upsampler (2×)        │
│  Conv → PixelShuffle(2)             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Reconstruction Head                │
│  Conv3×3 → LeakyReLU → Conv3×3(3)   │
└─────────────────────────────────────┘
      │
      ▼
Residual Detail  ──────────┐
                           ▼
Bicubic Baseline  ──────▶  +  ───▶  Output (3, 2H, 2W)  [Clamped 0-1]
```

### Key Components

| Component | Purpose | Complexity |
|-----------|---------|------------|
| **ResidualCNNBlock** | Local texture extraction with Channel Attention (SE) | O(C²) |
| **MDTA** | Multi-Dconv Head Transposed Attention — cross-channel attention | **O(H·W)** |
| **GFFN** | Gated Feed-Forward Network — depthwise conv + GELU gating | O(H·W·C) |
| **HybridGroup** | CNN → Transformer → CNN fusion | O(H·W·C) |
| **PixelShuffle** | Sub-pixel convolution for artifact-free upsampling | O(H·W·C) |
| **Global Residual** | Bicubic + Learned Detail = High-fidelity SR | — |

### Model Specifications

| Parameter | Value |
|-----------|-------|
| **Parameters** | 545,007 (~0.55M) |
| **Scale Factor** | 2× (configurable to 4×) |
| **Base Channels** | 48 |
| **Hybrid Groups** | 3 |
| **Attention Heads** | 4 |
| **Attention Type** | MDTA (linear O(H·W)) |
| **Activation** | LeakyReLU (0.1) / GELU |
| **Normalization** | GroupNorm (1 group = LayerNorm) |
| **Output Clamping** | [0, 1] |
| **Weight Initialization** | Kaiming Normal + Zero-residual final layer |

---

## 📡 API Reference

### Base URL
```
FastAPI:    http://localhost:8000
Vite Dev:   http://localhost:8080  (proxies /api → FastAPI)
```

### Endpoints

#### `GET /api/health`
Health check & engine status.

**Response**
```json
{
  "status": "online",
  "engine": "OrbitalHybridNet",
  "type": "CNN + Transformer Hybrid",
  "device": "CPU",
  "cuda_available": false,
  "parameter_count": 545007,
  "default_scale": 2,
  "message": "Orbital enhancement hybrid engine operational."
}
```

#### `GET /api/model/info`
Detailed model architecture information.

**Response**
```json
{
  "model_name": "OrbitalHybridNet",
  "version": "1.0",
  "total_parameters": 545007,
  "scale_factor": 2,
  "shallow_cnn_layers": 2,
  "hybrid_groups": 3,
  "attention_type": "MDTA (Multi-Dconv Head Transposed Self-Attention)",
  "attention_complexity": "O(H * W) Linear",
  "local_blocks": "Residual Channel Attention Blocks (RCAB)",
  "feed_forward": "Gated Feed-Forward Network (GFFN with GELU)",
  "reconstruction": "PixelShuffle Sub-Pixel Convolution",
  "global_residual_learning": true,
  "supported_inputs": ["PNG", "JPEG", "WEBP", "TIFF (RGB)"],
  "metrics_computed": ["PSNR", "SSIM", "RMSE", "MAE"]
}
```

#### `POST /api/enhance`
Enhance an orbital image with the hybrid model.

**Form Data**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `UploadFile` | **required** | Image file (PNG/JPG/WEBP/TIFF) |
| `scale` | `int` | `2` | Super-resolution factor (2 or 4) |
| `apply_dehaze` | `bool` | `true` | CLAHE atmospheric correction |
| `apply_sharpen` | `bool` | `true` | Edge sharpening for structures |
| `denoise_level` | `float` | `0.5` | Bilateral denoising strength (0-1) |
| `remove_clouds` | `bool` | `false` | Cloud detection + inpainting |
| `remove_obstacles` | `bool` | `false` | Dark/bright artifact removal |
| `deblur` | `bool` | `false` | Wiener-style deblurring |

**Response**
```json
{
  "status": "success",
  "enhanced_image": "data:image/png;base64,...",
  "original_image": "data:image/png;base64,...",
  "metrics": {
    "psnr": 28.45,
    "ssim": 0.8921,
    "rmse": 18.32,
    "mae": 14.17
  },
  "processing_time_ms": 1420.5,
  "input_resolution": [512, 512],
  "output_resolution": [1024, 1024],
  "device": "CPU",
  "model_architecture": "OrbitalHybridNet (CNN + MDTA Transformer)",
  "scale_factor": 2
}
```

#### `GET /api/samples`
List available demo samples.

#### `GET /samples/{filename}`
Serve static sample images.

---

## 🖥️ Frontend Features

### Interactive Workspace
- **Drag & drop** or browse for orbital imagery (PNG/JPG/WEBP/TIFF ≤20MB)
- **Preset samples**: Airport, Agricultural, Coastal
- **Real-time configuration**: Toggle CLAHE, sharpening, deblurring, cloud/obstacle removal
- **Live Three.js globe** visualization (WebGL)

### Results Viewer
- **Split slider** comparison (before/after)
- **Tabbed views**: Split / Enhanced / Original
- **Metadata bar**: Resolution, processing time, device, architecture
- **Scientific metrics panel**: PSNR, SSIM, RMSE, MAE with progress bars
- **One-click download** of enhanced PNG

### Authentication (Demo)
- Login / Signup modals (client-side only, no backend auth)

---

## 🧪 Training & Fine-Tuning

### Prepare Custom Dataset
```bash
# Place your HR satellite images in a folder
# Then generate augmented patches:
python prepare_dataset.py
# Creates 512×512 patches in ./dataset/
```

### Train from Scratch / Fine-Tune
```bash
# Basic training on custom dataset
python backend/train.py \
  --data_dir ./dataset/satellite_images \
  --epochs 25 \
  --batch_size 8 \
  --lr 2e-4 \
  --scale 2 \
  --patch_size 128

# Resume from checkpoint (auto-loads latest .pth)
python backend/train.py --data_dir ./my_data --epochs 50
```

### Training Details
- **Loss**: Charbonnier (smooth L1) — robust for image restoration
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing to 1e-6
- **Gradient Clipping**: max_norm=0.5
- **Mixed Precision**: FP16 on CUDA via `torch.compile` + `model.half()`

---

## 🌐 Deployment

### Docker (Recommended)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "run.py"]
```

```bash
docker build -t terra .
docker run -p 8000:8000 terra
```

### Cloud Platforms

| Platform | Notes |
|----------|-------|
| **Hugging Face Spaces** | Free GPU, `Dockerfile` or `requirements.txt` + `app.py` |
| **Railway** | `railway up` — auto-detects Python, free tier |
| **Render** | Web Service, set `pip install -r backend/requirements.txt && python run.py` |
| **Fly.io** | `fly launch` — GPU machines available |
| **Google Cloud Run** | `gcloud run deploy` — pay-per-request, scales to zero |

### Production Checklist
- [ ] `reload=False` in `run.py` (already set)
- [ ] Build frontend: `npm run build` (outputs to `dist/`)
- [ ] Serve static files via FastAPI `StaticFiles` (already configured)
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` for GPU memory
- [ ] Add reverse proxy (nginx/Traefik) for TLS termination
- [ ] Configure CORS origins in `app.py` (currently `allow_origins=["*"]`)

---

## 🔧 Configuration

### Environment Variables
```bash
# Optional: override default device
export DEVICE=cuda  # or cpu

# Optional: custom weights path
export WEIGHTS_PATH=/path/to/orbital_hybrid_net.pth
```

### Model Config (in `hybrid_transformer.py`)
```python
OrbitalHybridNet(
    in_channels=3,      # RGB
    out_channels=3,
    dim=48,             # base channels
    num_groups=3,       # hybrid groups
    num_heads=4,        # attention heads
    scale=2             # 2x or 4x
)
```

### Frontend Config (`vite.config.ts`)
```typescript
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', changeOrigin: true },
    '/samples': { target: 'http://localhost:8000', changeOrigin: true },
  }
}
```

---

## 📊 Performance Benchmarks

| Input Size | Device | Inference Time | Memory |
|------------|--------|----------------|--------|
| 256×256 → 512×512 | CPU (Intel i7) | ~1.4s | ~2 GB |
| 512×512 → 1024×1024 | CPU | ~4.2s | ~3 GB |
| 256×256 → 512×512 | GPU (RTX 3080) | ~85ms | ~1.5 GB VRAM |
| 512×512 → 1024×1024 | GPU (RTX 3080) | ~220ms | ~2.2 GB VRAM |

> **Note**: First request includes model warm-up. Subsequent requests are faster due to `torch.compile` caching.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style
- Python: `black` + `isort` (run `npm run format.fix`)
- TypeScript/JS: `prettier` (run `npm run format.fix`)
- Type checking: `npm run typecheck`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Restormer** (MDTA/GFFN) — Zamir et al., CVPR 2022
- **ESRGAN** (Residual-in-Residual, PixelShuffle) — Wang et al.
- **SwinIR** (Hybrid CNN-Transformer) — Liang et al.
- **Three.js** — Globe visualization
- **Radix UI / Tailwind CSS** — Frontend components & styling
- **Sentinel-2 / Landsat / SpaceNet** — Inspiration for orbital imagery domains

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: terra@example.com

---

<p align="center">
  <b>Terra — Image intelligence, elevated.</b><br>
  <sub>Built for a clearer future ✦</sub>
</p>