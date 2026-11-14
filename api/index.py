"""
DefectVision 2.0 - Vercel Serverless Function Entrypoint
Bridges FastAPI application with Vercel's Serverless Python runtime.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root directory to sys.path so modules (src, models, api_server) can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe fallback directories for serverless read-only environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

# Import the existing production FastAPI app
from api_server import app

# Expose app for Vercel Serverless runtime
__all__ = ["app"]
