FROM python:3.11-slim

WORKDIR /app

# HTTPS root certs for the Binance + HF Hub calls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Ensure /data exists and is writable regardless of the Space's runtime user.
# (HF persistent storage, if enabled, mounts over this; otherwise it's ephemeral
# and durability comes from the HF Hub dataset repo — see core/hf_store.py.)
RUN mkdir -p /data && chmod 777 /data

# HF Spaces persistent storage is mounted at /data
# Fallback: we also use HF Hub dataset repo for persistence
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
# Bind to $PORT if the platform sets one (Koyeb/Render/Fly), else 7860 (HF Spaces).
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
