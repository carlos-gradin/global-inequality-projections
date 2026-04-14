# ---------------------------------------------------------------------------
# Dockerfile for the Global Inequality Projections Streamlit app.
#
# Build:   docker build -t global-ineq-projections .
# Run:     docker run --rm -p 8501:8501 global-ineq-projections
# Browse:  http://localhost:8501
#
# This image is self-contained: all input parquets live inside it, so the
# app runs identically on any host that can run Docker — your laptop,
# a cloud VM, Hugging Face Spaces, Render, Fly.io, AWS/GCP/Azure, a
# university server, etc.  Zero code changes needed to move hosts.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# Unbuffered stdout/stderr so container logs stream live; no .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first so this layer is cached across code changes.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and its precomputed parquets.
COPY app/ ./app/

EXPOSE 8501

# --server.address=0.0.0.0 is required inside a container so the port is
# reachable from outside; --server.headless skips the browser auto-open
# (there's no browser inside the container).
CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
