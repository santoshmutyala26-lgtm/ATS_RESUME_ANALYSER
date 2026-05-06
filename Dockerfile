# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# System deps needed for some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch first (much smaller: ~300 MB vs 2 GB CUDA) ───────
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    flask-cors==4.0.1 \
    pdfminer.six==20231228 \
    pypdf==4.3.1 \
    spacy==3.7.5 \
    sentence-transformers==3.0.1 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    gunicorn==21.2.0

# ── Download spaCy English model ──────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Pre-download sentence-transformers model at build time ───────────────────
# This bakes the model into the image so first-request is fast
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model downloaded successfully')"

# ── Copy app source ───────────────────────────────────────────────────────────
COPY . .

# ── Expose port & run ─────────────────────────────────────────────────────────
EXPOSE 10000

# Use gunicorn for production (single worker to stay within free tier RAM)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]
