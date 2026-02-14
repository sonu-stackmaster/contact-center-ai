# CPU-Optimized Dockerfile for low-spec systems
FROM python:3.12-slim

# Set environment variables for CPU optimization
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies with CPU optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data cache logs

# Download required models (lightweight)
RUN python -c "
import nltk
from sentence_transformers import SentenceTransformer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Pre-download sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print('Models downloaded successfully')
"

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command (optimized for CPU)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]