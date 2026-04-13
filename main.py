# -*- coding: utf-8 -*-
"""
main.py – AI Müzik & Ses Prodüksiyon Sistemi – Ana Giriş Noktası
=================================================================
Bu dosya sistemi iki modda başlatır:

    python main.py api       → FastAPI REST sunucusu (port 8000)
    python main.py ui        → Streamlit web arayüzü (port 8501)
    python main.py pipeline  → Komut satırından tek seferlik pipeline

Kullanım örnekleri:
    # API sunucusu
    python main.py api

    # Streamlit arayüzü
    python main.py ui

    # Komut satırı pipeline
    python main.py pipeline \
        --text "Türkiye'nin en iyi akıllı telefon markası." \
        --tone enerjik \
        --duration 30 \
        --output cikti.wav
"""

# ──────────────────────────────────────────────
# UTF-8 STDOUT – Türkçe çıktı için zorunlu
# ──────────────────────────────────────────────
import sys
import os

# Python 3.7+ için stdout/stderr encoding zorla
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
if sys.stderr.encoding != "utf-8":
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
import logging
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ──────────────────────────────────────────────
# PROJE YOLU
# ──────────────────────────────────────────────
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import API_CONFIG, OUTPUTS_DIR
from utils.turkish_utils import setup_turkish_locale

# Sistem locale'ini Türkçe'ye ayarla
setup_turkish_locale()


# ──────────────────────────────────────────────
# MOD: API SUNUCUSU
# ──────────────────────────────────────────────
def run_api():
    """FastAPI sunucusunu başlatır."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn kurulu değil. Çalıştırın: pip install uvicorn")
        sys.exit(1)

    from api.routes import app
    logger.info("API sunucusu başlatılıyor: http://%s:%d",
                API_CONFIG["host"], API_CONFIG["port"])
    uvicorn.run(
        app,
        host    = API_CONFIG["host"],
        port    = API_CONFIG["port"],
        reload  = API_CONFIG["debug"],
        log_level = "info",
    )


# ──────────────────────────────────────────────
# MOD: STREAMLIT ARAYÜZÜ
# ──────────────────────────────────────────────
def run_ui():
    """Streamlit arayüzünü başlatır."""
    import subprocess
    app_path = ROOT / "app.py"
    logger.info("Streamlit arayüzü başlatılıyor…")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
    ])


# ──────────────────────────────────────────────
# MOD: KOMUt SATIRI PİPELINE
# ──────────────────────────────────────────────
def run_pipeline(args):
    """
    Tek seferlik tam prodüksiyon pipeline'ı.

    Kullanım:
        python main.py pipeline --text "..." --tone enerjik --duration 30 --output out.wav
    """
    from models import VoiceEngine, MusicGenerator, VoiceCloner
    from utils  import AudioMixer, TextAnalyzer, FileManager
    from utils.turkish_utils import TurkishUtils

    text         = args.text
    tone_profile = args.tone
    duration     = args.duration
    output_path  = args.output or str(OUTPUTS_DIR / "pipeline_output.wav")
    voice_id     = args.voice_id

    logger.info("Pipeline başlıyor")
    logger.info("Metin    : %s", text[:80])
    logger.info("Ton      : %s", tone_profile)
    logger.info("Süre     : %ds", duration)
    logger.info("Çıktı    : %s", output_path)

    # Bileşenler
    voice_engine    = VoiceEngine()
    music_generator = MusicGenerator()
    voice_cloner    = VoiceCloner()
    mixer           = AudioMixer()
    analyzer        = TextAnalyzer()
    file_manager    = FileManager()

    # Metin normalleştir
    normalized = TurkishUtils.normalize_for_tts(text)
    analysis   = analyzer.full_analysis(normalized)

    if not tone_profile or tone_profile == "auto":
        tone_profile = analysis["tone"]
        logger.info("Otomatik ton: %s (güven: %.2f)", tone_profile, analysis["confidence"])

    music_prompt = analysis["music_prompt"]

    # ① TTS
    logger.info("Adım 1/3: Ses sentezi…")
    voice_path  = file_manager.new_output_path("voice_raw")
    speaker_wav = None
    if voice_id:
        speaker_wav = voice_cloner.get_voice_path(voice_id)
    voice_engine.synthesize(
        text         = normalized,
        output_path  = voice_path,
        speaker_wav  = speaker_wav,
        tone_profile = tone_profile,
    )
    logger.info("  → Ses: %s", voice_path)

    # ② Müzik
    logger.info("Adım 2/3: Müzik üretimi…")
    music_path = file_manager.new_output_path("music_raw")
    music_generator.generate(
        prompt       = music_prompt,
        output_path  = music_path,
        duration     = duration,
        tone_profile = tone_profile,
    )
    logger.info("  → Müzik: %s", music_path)

    # ③ Mix
    logger.info("Adım 3/3: Mix & Mastering…")
    mixer.mix(
        voice_path  = voice_path,
        music_path  = music_path,
        output_path = output_path,
    )
    logger.info("Pipeline tamamlandı → %s", output_path)

    # Ara dosyalar temizle
    for p in [voice_path, music_path]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    print(f"\nNihai çıktı: {output_path}")
    return output_path


# ──────────────────────────────────────────────
# GİRİŞ NOKTASI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Müzik & Ses Prodüksiyon Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py api
  python main.py ui
  python main.py pipeline --text "Harika bir ürün!" --tone enerjik --duration 30
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # api
    subparsers.add_parser("api", help="FastAPI sunucusunu başlat")

    # ui
    subparsers.add_parser("ui", help="Streamlit arayüzünü başlat")

    # pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Komut satırı pipeline")
    pipe_parser.add_argument("--text",      required=True,  help="Seslendirme metni")
    pipe_parser.add_argument("--tone",      default="auto", help="Reklam tonu (auto|enerjik|kurumsal|dramatik|samimi|lüks)")
    pipe_parser.add_argument("--duration",  type=int, default=30, help="Müzik süresi (saniye)")
    pipe_parser.add_argument("--output",    default=None,   help="Çıktı dosya yolu")
    pipe_parser.add_argument("--voice-id",  default=None,   dest="voice_id", help="Klonlama profil ID'si")

    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "pipeline":
        run_pipeline(args)
