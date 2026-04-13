# -*- coding: utf-8 -*-
"""
config.py – Merkezi Konfigürasyon Modülü
=========================================
Tüm ortam değişkenleri, yol sabitleri ve model parametreleri
bu dosyada tanımlanır. Değişiklikler yalnızca buradan yapılır.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle (varsa)
load_dotenv()

# ──────────────────────────────────────────────
# TEMEL DIZIN YAPISI
# ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
UPLOADS_DIR    = BASE_DIR / "uploads"
VOICES_DIR     = UPLOADS_DIR / "voices"     # Kullanıcı ses örnekleri
OUTPUTS_DIR    = UPLOADS_DIR / "outputs"    # Üretilen dosyalar
MODELS_DIR     = BASE_DIR / "models"
ASSETS_DIR     = BASE_DIR / "assets"

# Klasörleri otomatik oluştur
for _dir in [UPLOADS_DIR, VOICES_DIR, OUTPUTS_DIR, ASSETS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# SES MOTORU (TTS) AYARLARI
# ──────────────────────────────────────────────
TTS_CONFIG = {
    "model_name"    : os.getenv("TTS_MODEL", "tts_models/tr/common-voice/glow-tts"),  # Türkçe varsayılan
    "xtts_model"    : os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
    "language"      : "tr",          # Türkçe dil kodu
    "use_gpu"       : os.getenv("USE_GPU", "true").lower() == "true",
    "speaker_wav"   : None,          # Klonlama için referans ses (runtime'da atanır)
    "sample_rate"   : 24000,         # Hz – XTTS v2 çıkış örnekleme hızı
    "speed"         : 1.0,           # Konuşma hızı (1.0 = normal)
}

# ──────────────────────────────────────────────
# MÜZİK ÜRETİM AYARLARI (MusicGen / AudioLDM)
# ──────────────────────────────────────────────
MUSIC_CONFIG = {
    "model"         : os.getenv("MUSIC_MODEL", "facebook/musicgen-small"),
    "duration"      : 30,            # saniye – varsayılan müzik uzunluğu
    "sample_rate"   : 32000,         # Hz – MusicGen çıkış örnekleme hızı
    "guidance_scale": 3.0,           # Prompt'a uyum katsayısı
    "top_k"         : 250,
    "temperature"   : 1.0,
}

# ──────────────────────────────────────────────
# SES MİXLEME AYARLARI
# ──────────────────────────────────────────────
MIX_CONFIG = {
    "voice_volume_db"  : 0,          # dB – Konuşma ses seviyesi (0 = değişiklik yok)
    "music_volume_db"  : -12,        # dB – Müzik arka plan seviyesi
    "duck_volume_db"   : -6,         # dB – Ducking sırasında müzik seviyesi düşüşü
    "duck_attack_ms"   : 50,         # ms – Ducking başlama süresi
    "duck_release_ms"  : 200,        # ms – Ducking bitiş süresi
    "fade_in_ms"       : 500,        # ms – Giriş fade süresi
    "fade_out_ms"      : 1000,       # ms – Çıkış fade süresi
    "output_format"    : "wav",      # wav | mp3
    "output_sample_rate": 44100,     # Hz – Nihai çıkış
}

# ──────────────────────────────────────────────
# REKLAM TONU PROFİLLERİ
# ──────────────────────────────────────────────
AD_TONE_PROFILES = {
    "enerjik": {
        "speed"       : 1.15,
        "description" : "Hızlı tempolu, dinamik ve coşkulu reklam tonu",
        "music_prompt": "upbeat energetic electronic music, fast tempo, exciting",
    },
    "kurumsal": {
        "speed"       : 0.92,
        "description" : "Güven veren, profesyonel ve otoriter kurumsal ton",
        "music_prompt": "corporate background music, professional, calm piano strings",
    },
    "dramatik": {
        "speed"       : 0.85,
        "description" : "Duygusal, güçlü ve etkileyici dramatik ton",
        "music_prompt": "dramatic orchestral music, emotional, cinematic, epic strings",
    },
    "samimi": {
        "speed"       : 1.0,
        "description" : "Sıcak, arkadaşça ve doğal konuşma tonu",
        "music_prompt": "warm friendly acoustic music, gentle guitar, natural feel",
    },
    "lüks": {
        "speed"       : 0.88,
        "description" : "Zarif, sofistike ve premium ürün tonu",
        "music_prompt": "elegant luxury music, jazz piano, sophisticated, smooth",
    },
}

# ──────────────────────────────────────────────
# API AYARLARI
# ──────────────────────────────────────────────
API_CONFIG = {
    "host"          : os.getenv("API_HOST", "0.0.0.0"),
    "port"          : int(os.getenv("API_PORT", 8000)),
    "debug"         : os.getenv("API_DEBUG", "false").lower() == "true",
    "max_upload_mb" : int(os.getenv("MAX_UPLOAD_MB", 100)),
    "allowed_audio" : {".wav", ".mp3", ".ogg", ".flac", ".m4a"},
    "secret_key"    : os.getenv("SECRET_KEY", "change-me-in-production"),
}

# ──────────────────────────────────────────────
# STREAMLIT ARAYÜZ AYARLARI
# ──────────────────────────────────────────────
UI_CONFIG = {
    "page_title"    : "AI Müzik & Ses Prodüksiyon Stüdyosu",
    "page_icon"     : "🎵",
    "layout"        : "wide",
    "theme_primary" : "#6C63FF",
}

# ──────────────────────────────────────────────
# TÜRKÇE KARAKTER KODLAMA
# ──────────────────────────────────────────────
ENCODING_CONFIG = {
    "default"       : "utf-8",
    "locale"        : "tr_TR.UTF-8",
    "turkish_chars" : "çÇğĞıİöÖşŞüÜ",
}
