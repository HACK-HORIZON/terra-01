"""
Root Application Launcher for Orbital Image Intelligence Platform.
Boots the FastAPI backend serving both the CNN-Transformer Hybrid Engine
and the interactive web frontend on http://localhost:8000.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  SPHERE: Orbital Image Intelligence Platform")
    print("  Hybrid CNN + Transformer Deep Learning Super-Resolution")
    print("=" * 60)
    print("Starting server at http://localhost:8000 ...")
    print("Press Ctrl+C to stop.")

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
