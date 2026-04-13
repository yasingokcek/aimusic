# ══════════════════════════════════════════════
# Dockerfile – AI Müzik & Ses Prodüksiyon Sistemi
# ══════════════════════════════════════════════
# Temel imaj: NVIDIA CUDA 11.8 + cuDNN 8 (GPU desteği)
# Python 3.10 üzerinde çalışır.
#
# Build: docker build -t ai-music-prod .
# Run  : docker run --gpus all -p 8000:8000 -p 8501:8501 ai-music-prod
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# AŞAMA 1 – TEMEL İMAJ (GPU)
# GPU kullanmıyorsanız: python:3.10-slim ile değiştirin
# ──────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# ──────────────────────────────────────────────
# SİSTEM BAĞIMLILIKLARI
# Türkçe locale + FFmpeg (ses işleme) + build araçları
# ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=tr_TR.UTF-8 \
    LANGUAGE=tr_TR:tr \
    LC_ALL=tr_TR.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        # Türkçe dil paketi
        language-pack-tr \
        locales \
        # Ses araçları
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        # Python 3.10
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        # Derleme araçları
        build-essential \
        git \
        wget \
        curl \
        # Ağ araçları
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Türkçe locale oluştur
RUN locale-gen tr_TR.UTF-8 && update-locale LANG=tr_TR.UTF-8

# Python 3.10'u varsayılan yap
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

# ──────────────────────────────────────────────
# AŞAMA 2 – PYTHON BAĞIMLILIKLARI
# ──────────────────────────────────────────────
FROM base AS dependencies

WORKDIR /app

# pip güncelle
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 11.8) – önce kur, büyük katman ayrı kalır
RUN pip install torch==2.3.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu118

# Diğer bağımlılıklar
COPY requirements.txt .
RUN pip install -r requirements.txt

# ──────────────────────────────────────────────
# AŞAMA 3 – UYGULAMA
# ──────────────────────────────────────────────
FROM dependencies AS app

WORKDIR /app

# Uygulama kaynak kodunu kopyala
COPY . .

# Gerekli dizinleri oluştur
RUN mkdir -p uploads/voices uploads/outputs assets/samples

# ──────────────────────────────────────────────
# PORT AÇMA
# ──────────────────────────────────────────────
# 8000 → FastAPI REST API
# 8501 → Streamlit Web Arayüzü
EXPOSE 8000 8501

# ──────────────────────────────────────────────
# SAĞLIK KONTROLÜ
# ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ──────────────────────────────────────────────
# BAŞLATMA KOMUTU
# Varsayılan: API sunucusu
# UI için: docker run ... python main.py ui
# ──────────────────────────────────────────────
CMD ["python", "main.py", "api"]
