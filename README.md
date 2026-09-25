# TERRA-01: Orbital Image Intelligence Platform

Hybrid CNN + Transformer Super-Resolution (2x) and Satellite Image Enhancement Engine.

---

## 🚀 Quick Setup for Collaborators

### 1. Clone the Repository
```bash
git clone https://github.com/Priyangshu6146/terra-01.git
cd terra-01
```

### 2. Install Dependencies
Make sure you have **Python 3.10, 3.11, or 3.12** installed.

#### Option A: GPU (NVIDIA CUDA - Recommended for fast inference)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

#### Option B: CPU Only
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application Locally

Start the local server (FastAPI backend + Interactive web UI):
```bash
python run.py
```
Open your browser and navigate to:
👉 **http://localhost:8000**

---

## 🛠️ Project Structure

```
terra-01/
├── run.py                          # Root launcher (runs FastAPI + UI on localhost:8000)
├── app.py                          # Cloud deployment root wrapper
├── requirements.txt                # Root requirements pointing to backend dependencies
├── render.yaml                     # Render deployment configuration
├── backend/
│   ├── app.py                      # FastAPI REST API endpoints (/api/enhance, /api/health)
│   ├── enhancer.py                 # Enhancement engine (Hybrid inference + LAB edge restoration)
│   ├── train.py                    # Edge-aware training script for fine-tuning weights
│   ├── requirements.txt            # Backend Python dependencies
│   └── model/
│       ├── hybrid_transformer.py   # OrbitalHybridNet (CNN + MDTA Transformer architecture)
│       ├── metrics.py              # Scientific metrics calculation (PSNR, SSIM, RMSE, MAE)
│       └── weights/
│           └── orbital_hybrid_net.pth  # Trained model weights checkpoint
└── Frontend/
    ├── index.html                  # Web application UI (Comparison slider, 1:1 zoom mode)
    ├── styles.css                  # Responsive dark mode styling
    ├── script.js                   # Client-side logic, API connection, and slider controls
    └── public/
        └── samples/                # Preset satellite test images
```

---

## 🎯 Fine-Tuning the Model (Optional)

To continue training or fine-tuning the model with high-frequency EdgeLoss:
```bash
python backend/train.py --epochs 25 --lr 0.0003 --batch_size 8
```
Model weights are automatically saved to `backend/model/weights/orbital_hybrid_net.pth`.

---

## 🔄 Deployment Workflow

- **`main` branch** is connected to automated deployment pipelines:
  - **Render**: Automatically deploys the FastAPI backend container on git push.
  - **Vercel**: Automatically deploys the frontend web app on git push.
- When you are ready to ship your changes:
  ```bash
  git add .
  git commit -m "feat/fix: description of your changes"
  git push origin main
  ```
