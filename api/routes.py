# -*- coding: utf-8 -*-
"""
routes.py – FastAPI REST API Endpoint'leri
==========================================
Sisteme HTTP üzerinden erişim sağlar.
Streamlit arayüzü veya harici entegrasyonlar bu API'yi kullanabilir.

Endpoint'ler:
    POST /synthesize          → Metni sese çevir
    POST /generate-music      → Müzik üret
    POST /mix                 → Ses + Müzik karıştır
    POST /full-pipeline       → Tam pipeline (metin → nihai ses)
    POST /voices/register     → Ses profili kaydet
    GET  /voices              → Kayıtlı sesleri listele
    DELETE /voices/{voice_id} → Ses profilini sil
    GET  /outputs             → Üretilen dosyaları listele
    GET  /download/{filename} → Dosya indir
    GET  /health              → Sistem durumu

Bağımlılıklar:
    pip install fastapi uvicorn python-multipart aiofiles
"""

import os
import logging
from pathlib import Path
from typing  import Optional

from fastapi           import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Proje modülleri
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config           import API_CONFIG, OUTPUTS_DIR
from models           import VoiceEngine, MusicGenerator, VoiceCloner
from utils            import AudioMixer, TextAnalyzer, FileManager
from utils.turkish_utils import TurkishUtils

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FASTAPI UYGULAMA
# ──────────────────────────────────────────────
app = FastAPI(
    title       = "AI Müzik & Ses Prodüksiyon API",
    description = "Türkçe TTS, Ses Klonlama ve AI Müzik Üretimi",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS – Streamlit ve harici istemciler için
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Production'da kısıtla
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Bileşen örnekleri (singleton – uygulama ömrü boyunca)
_voice_engine    = VoiceEngine()
_music_generator = MusicGenerator()
_voice_cloner    = VoiceCloner()
_mixer           = AudioMixer()
_text_analyzer   = TextAnalyzer()
_file_manager    = FileManager()

# ──────────────────────────────────────────────
# PYDANTIC ŞEMALARI (Request/Response modelleri)
# ──────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text         : str
    voice_id     : Optional[str] = None   # Klonlama için
    tone_profile : str           = "kurumsal"
    language     : str           = "tr"
    speed        : Optional[float] = None


class MusicRequest(BaseModel):
    prompt       : str
    duration     : int           = 30
    tone_profile : Optional[str] = None


class MixRequest(BaseModel):
    voice_file   : str           # Sunucu tarafındaki yol
    music_file   : str           # Sunucu tarafındaki yol
    ducking      : bool          = True
    duck_ratio   : float         = 0.30


class FullPipelineRequest(BaseModel):
    text         : str
    voice_id     : Optional[str] = None
    tone_profile : Optional[str] = None   # None → otomatik tespit
    music_prompt : Optional[str] = None   # None → otomatik oluştur
    duration     : Optional[int] = None   # None → metin süresine göre


# ──────────────────────────────────────────────
# SISTEM DURUMU
# ──────────────────────────────────────────────
@app.get("/health", tags=["Sistem"])
async def health_check():
    """Sistem sağlık kontrolü."""
    import torch
    return {
        "status"    : "ok",
        "gpu"       : torch.cuda.is_available(),
        "gpu_name"  : torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "voices"    : len(_voice_cloner.list_voices()),
        "outputs"   : len(_file_manager.list_outputs()),
    }


# ──────────────────────────────────────────────
# SES SENTEZİ
# ──────────────────────────────────────────────
@app.post("/synthesize", tags=["TTS"])
async def synthesize(req: SynthesizeRequest):
    """
    Türkçe metni sese çevirir.

    - voice_id varsa: klonlama modu
    - tone_profile: enerjik | kurumsal | dramatik | samimi | lüks
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    normalized_text = TurkishUtils.normalize_for_tts(req.text)
    output_path     = _file_manager.new_output_path("voice")

    speaker_wav = None
    if req.voice_id:
        try:
            speaker_wav = _voice_cloner.get_voice_path(req.voice_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Ses profili bulunamadı: {req.voice_id}")

    try:
        result = _voice_engine.synthesize(
            text         = normalized_text,
            output_path  = output_path,
            speaker_wav  = speaker_wav,
            language     = req.language,
            tone_profile = req.tone_profile,
            speed        = req.speed,
        )
    except Exception as exc:
        logger.error("TTS hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    filename = Path(result).name
    return {
        "status"    : "ok",
        "file"      : filename,
        "path"      : result,
        "download"  : f"/download/{filename}",
    }


# ──────────────────────────────────────────────
# MÜZİK ÜRETİMİ
# ──────────────────────────────────────────────
@app.post("/generate-music", tags=["Müzik"])
async def generate_music(req: MusicRequest):
    """Metin tanımından arka plan müziği üretir."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Müzik prompt'u boş olamaz.")

    output_path = _file_manager.new_output_path("music")

    try:
        result = _music_generator.generate(
            prompt       = req.prompt,
            output_path  = output_path,
            duration     = req.duration,
            tone_profile = req.tone_profile,
        )
    except Exception as exc:
        logger.error("Müzik üretim hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    filename = Path(result).name
    return {
        "status"  : "ok",
        "file"    : filename,
        "path"    : result,
        "download": f"/download/{filename}",
    }


# ──────────────────────────────────────────────
# SES + MÜZİK MİX
# ──────────────────────────────────────────────
@app.post("/mix", tags=["Prodüksiyon"])
async def mix_audio(req: MixRequest):
    """Ses ve müziği karıştırır (ducking ile)."""
    for path_str in [req.voice_file, req.music_file]:
        if not Path(path_str).exists():
            raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {path_str}")

    output_path = _file_manager.new_output_path("final_mix")

    try:
        result = _mixer.mix(
            voice_path  = req.voice_file,
            music_path  = req.music_file,
            output_path = output_path,
            ducking     = req.ducking,
            duck_ratio  = req.duck_ratio,
        )
    except Exception as exc:
        logger.error("Mix hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    filename = Path(result).name
    return {
        "status"  : "ok",
        "file"    : filename,
        "path"    : result,
        "download": f"/download/{filename}",
    }


# ──────────────────────────────────────────────
# TAM PIPELINE (Metin → Nihai Ses)
# ──────────────────────────────────────────────
@app.post("/full-pipeline", tags=["Prodüksiyon"])
async def full_pipeline(req: FullPipelineRequest, background_tasks: BackgroundTasks):
    """
    Tek istekle tam prodüksiyon:
    Metin → TTS → AI Müzik → Mix → Nihai .wav

    Adımlar:
    1. Metin analizi (ton + kategori)
    2. TTS ile ses üret
    3. Müzik üret
    4. Mix + Ducking
    5. Dosyayı döner
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    normalized_text = TurkishUtils.normalize_for_tts(req.text)

    # ① Metin analizi
    analysis     = _text_analyzer.full_analysis(normalized_text)
    tone_profile = req.tone_profile or analysis["tone"]
    music_prompt = req.music_prompt or analysis["music_prompt"]
    duration     = req.duration     or max(30, int(analysis["stats"]["estimated_sec"]) + 10)

    # ② TTS
    voice_path   = _file_manager.new_output_path("voice_raw")
    speaker_wav  = None
    if req.voice_id:
        try:
            speaker_wav = _voice_cloner.get_voice_path(req.voice_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Ses profili bulunamadı: {req.voice_id}")

    _voice_engine.synthesize(
        text         = normalized_text,
        output_path  = voice_path,
        speaker_wav  = speaker_wav,
        tone_profile = tone_profile,
    )

    # ③ Müzik
    music_path = _file_manager.new_output_path("music_raw")
    _music_generator.generate(
        prompt       = music_prompt,
        output_path  = music_path,
        duration     = duration,
        tone_profile = tone_profile,
    )

    # ④ Mix
    final_path = _file_manager.new_output_path("final_mix")
    _mixer.mix(
        voice_path  = voice_path,
        music_path  = music_path,
        output_path = final_path,
    )

    # ⑤ Ara dosyaları arka planda temizle
    def _cleanup():
        for p in [voice_path, music_path]:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    background_tasks.add_task(_cleanup)

    filename = Path(final_path).name
    return {
        "status"        : "ok",
        "file"          : filename,
        "path"          : final_path,
        "download"      : f"/download/{filename}",
        "analysis"      : analysis,
        "tone_used"     : tone_profile,
        "music_prompt"  : music_prompt,
        "duration_sec"  : duration,
    }


# ──────────────────────────────────────────────
# SES PROFİLİ YÖNETİMİ
# ──────────────────────────────────────────────
@app.post("/voices/register", tags=["Ses Profilleri"])
async def register_voice(
    file        : UploadFile = File(...),
    name        : str        = Form(...),
    description : str        = Form(""),
):
    """Yüklenen ses dosyasını klonlama profili olarak kaydeder."""
    content = await file.read()

    try:
        temp_path = _file_manager.save_upload(content, file.filename)
        profile   = _voice_cloner.register_voice(
            audio_path  = temp_path,
            voice_name  = name,
            description = description,
        )
        Path(temp_path).unlink(missing_ok=True)   # Geçici dosyayı sil
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Profil kayıt hatası: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "profile": profile}


@app.get("/voices", tags=["Ses Profilleri"])
async def list_voices():
    """Kayıtlı ses profillerini listeler."""
    return {"voices": _voice_cloner.list_voices()}


@app.delete("/voices/{voice_id}", tags=["Ses Profilleri"])
async def delete_voice(voice_id: str):
    """Ses profilini siler."""
    deleted = _voice_cloner.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Profil bulunamadı.")
    return {"status": "ok", "deleted": voice_id}


# ──────────────────────────────────────────────
# DOSYA İNDİRME / LİSTELEME
# ──────────────────────────────────────────────
@app.get("/outputs", tags=["Dosyalar"])
async def list_outputs():
    """Üretilmiş çıktı dosyalarını listeler."""
    return {"outputs": _file_manager.list_outputs()}


@app.get("/download/{filename}", tags=["Dosyalar"])
async def download_file(filename: str):
    """Belirtilen dosyayı indirir."""
    # Güvenlik: path traversal önleme
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

    return FileResponse(
        path             = str(file_path),
        filename         = safe_name,
        media_type       = "audio/wav",
    )


# ──────────────────────────────────────────────
# METİN ANALİZİ (ön izleme için)
# ──────────────────────────────────────────────
@app.post("/analyze-text", tags=["Analiz"])
async def analyze_text(text: str = Form(...)):
    """Metni analiz ederek ton, kategori ve istatistikleri döner."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    normalized = TurkishUtils.normalize_for_tts(text)
    analysis   = _text_analyzer.full_analysis(normalized)
    return {"status": "ok", "analysis": analysis}
