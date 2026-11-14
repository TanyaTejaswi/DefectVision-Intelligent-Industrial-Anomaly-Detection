FROM python:3.10-slim

# Avoid interactive prompts and constrain multi-thread memory allocations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MPLCONFIGDIR=/tmp/matplotlib \
    LOKY_MAX_CPU_COUNT=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TORCH_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user matching Hugging Face Spaces & Render non-root requirements (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Pre-install lightweight CPU-only PyTorch & Torchvision (~180 MB instead of ~2.5 GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install application requirements
COPY --chown=user:user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, curated samples, and static frontend
COPY --chown=user:user . $HOME/app/

# Expose server port (Render & HF dynamically set PORT)
EXPOSE 7860

# Launch DefectVision FastAPI Server with single worker to keep memory < 200MB
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
