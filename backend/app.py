"""
FastAPI Backend Application for Orbital Image Intelligence Platform.
Provides RESTful APIs for CNN-Transformer Hybrid Super-Resolution & Enhancement,
scientific metric evaluation, and static asset serving.
"""

import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch

try:
    from .enhancer import get_enhancer
    from .model.hybrid_transformer import OrbitalHybridNet
except ImportError:
    from enhancer import get_enhancer
    from model.hybrid_transformer import OrbitalHybridNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Orbital Image Enhancer API",
    description="Deep Learning CNN + Transformer Hybrid Satellite Image Enhancement Engine",
    version="1.0.0"
)

# CORS configuration - allow all origins (including Vercel, localhost, previews)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("[API] Orbital Enhancement Engine starting...")
    import torch
    torch.set_num_threads(2)  # Prevent thread pool RAM explosion on cloud container
    get_enhancer()
    print("[API] Model pre-loaded and ready for fast inference.")


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "engine": "OrbitalHybridNet",
        "type": "CNN + Transformer Hybrid",
        "device": "CPU",
        "default_scale": 2,
        "message": "Orbital enhancement hybrid engine operational."
    }


@app.get("/api/model/info")
async def model_info():
    enhancer = get_enhancer()
    model = enhancer.model_2x
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "OrbitalHybridNet",
        "version": "1.0",
        "total_parameters": param_count,
        "scale_factor": model.scale,
        "shallow_cnn_layers": len(model.shallow_conv),
        "hybrid_groups": len(model.groups),
        "attention_type": "MDTA (Multi-Dconv Head Transposed Self-Attention)",
        "attention_complexity": "O(H * W) Linear",
        "local_blocks": "Residual Channel Attention Blocks (RCAB)",
        "feed_forward": "Gated Feed-Forward Network (GFFN with GELU)",
        "reconstruction": "PixelShuffle Sub-Pixel Convolution",
        "global_residual_learning": True,
        "supported_inputs": ["PNG", "JPEG", "WEBP", "TIFF (RGB)"],
        "metrics_computed": ["PSNR", "SSIM", "RMSE", "MAE"]
    }


@app.post("/api/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    scale: int = Form(2),
    apply_dehaze: bool = Form(True),
    apply_sharpen: bool = Form(True),
    denoise_level: float = Form(0.5),
    remove_clouds: bool = Form(False),
    remove_obstacles: bool = Form(False),
    deblur: bool = Form(False)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        import gc
        image_bytes = await file.read()
        enhancer = get_enhancer()
        result = enhancer.enhance_image(
            image_bytes=image_bytes,
            scale=scale,
            apply_dehaze=apply_dehaze,
            apply_sharpen=apply_sharpen,
            denoise_level=denoise_level,
            remove_clouds=remove_clouds,
            remove_obstacles=remove_obstacles,
            deblur=deblur
        )
        del image_bytes
        gc.collect()
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhancement error: {str(e)}")
    finally:
        import gc
        gc.collect()


@app.get("/api/samples")
async def list_samples():
    samples_dir = os.path.join(BASE_DIR, "public", "samples")
    samples = [
        {
            "id": "airport",
            "name": "Urban Airfield",
            "description": "Commercial airport runways, terminal apron, and taxiways",
            "url": "/samples/sample_airport.jpg"
        },
        {
            "id": "agricultural",
            "name": "Agricultural Swaths",
            "description": "Circular center-pivot irrigation crops and farmland grids",
            "url": "/samples/sample_agricultural.jpg"
        },
        {
            "id": "coastal",
            "name": "Coastal Harbour",
            "description": "Marine port with cargo ships, breakwater docks, and urban coast",
            "url": "/samples/sample_coastal.jpg"
        }
    ]
    return {"samples": samples}


# Serve static public assets (/samples, /favicon.ico, etc.)
public_path = os.path.join(BASE_DIR, "public")
if os.path.exists(public_path):
    app.mount("/public", StaticFiles(directory=public_path), name="public")
    app.mount("/samples", StaticFiles(directory=os.path.join(public_path, "samples")), name="samples")

# Serve root static assets (styles.css, script.js)
@app.get("/styles.css")
async def get_styles():
    return FileResponse(os.path.join(BASE_DIR, "styles.css"), media_type="text/css")

@app.get("/script.js")
async def get_script():
    return FileResponse(os.path.join(BASE_DIR, "script.js"), media_type="application/javascript")

# Root index page
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")
