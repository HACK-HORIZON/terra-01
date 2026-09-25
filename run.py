"""
Terra: Orbital Image Intelligence Platform Launcher.
Runs the FastAPI backend and serves the frontend on http://localhost:8000.
"""

import sys
import os
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    print("=" * 60)
    print("  TERRA: Orbital Image Intelligence Platform")
    print("  Hybrid CNN + Transformer Deep Learning Super-Resolution")
    print("=" * 60)
    print("Starting server at http://localhost:8000 ...")
    print("Press Ctrl+C to stop.")

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=ROOT_DIR
    )
