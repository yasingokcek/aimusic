# -*- coding: utf-8 -*-
"""
voice_engine.py – Türkçe TTS (Text-to-Speech) Motoru
======================================================
XTTS v2 (Coqui TTS) kullanarak Türkçe metni sese çevirir.
Ses klonlama, reklam tonu ayarı ve duygu bazlı sentez destekler.

Bağımlılıklar:
    pip install TTS torch torchaudio
"""

import sys
import logging
from pathlib import Path
from typing  import Optional

import torch
import numpy as np

# Proje kök dizini sys.path'e eklenir (import sorunlarını önler)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TTS_CONFIG, AD_TONE_PROFILES, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class VoiceEngine:
    """
    Türkçe TTS motoru.

    Özellikler:
    - XTTS v2 ile çok dilli, yüksek kaliteli ses sentezi
    - Referans ses dosyasından klonlama (zero-shot voice cloning)
    - Reklam tonu profiline göre hız/enerji ayarı
    - GPU/CPU otomatik seçimi
    """

    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Parametreler
        ------------
        use_gpu : bool, isteğe bağlı
            GPU kullanılsın mı? None ise config.py'daki ayar kullanılır.
        """
        self.use_gpu = use_gpu if use_gpu is not None else TTS_CONFIG["use_gpu"]
        self.device  = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        self.model   = None          # TTS nesnesi (lazy loading)
        self._loaded = False

        logger.info("VoiceEngine başlatıldı | Cihaz: %s", self.device)

    # ──────────────────────────────────────────
    # MODEL YÜKLEME
    # ──────────────────────────────────────────
    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        XTTS v2 modelini belleğe yükler.
        İlk çağrıda model indirilir (~1.8 GB).

        Parametreler
        ------------
        model_name : str, isteğe bağlı
            HuggingFace / Coqui model kimliği.
            Varsayılan: config.py TTS_CONFIG["xtts_model"]
        """
        if self._loaded:
            logger.debug("Model zaten yüklü, atlıyorum.")
            return

        # Coqui TTS kütüphanesi lazy import edilir (kurulu olmayabilir)
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS kurulu değil. Lütfen çalıştırın: pip install TTS"
            ) from exc

        _model = model_name or TTS_CONFIG["xtts_model"]
        logger.info("TTS modeli yükleniyor: %s", _model)

        # gpu=True → CUDA; gpu=False → CPU
        self.model   = CoquiTTS(model_name=_model, gpu=self.use_gpu)
        self._loaded = True
        logger.info("TTS modeli başarıyla yüklendi.")

    # ──────────────────────────────────────────
    # SES ÜRETME
    # ──────────────────────────────────────────
    def synthesize(
        self,
        text         : str,
        output_path  : str,
        speaker_wav  : Optional[str] = None,
        language     : str           = "tr",
        tone_profile : str           = "kurumsal",
        speed        : Optional[float] = None,
    ) -> str:
        """
        Metni sese çevirir ve dosyaya kaydeder.

        Parametreler
        ------------
        text         : Seslendirilecek Türkçe metin.
        output_path  : Çıktı .wav dosya yolu.
        speaker_wav  : Referans ses dosyası (klonlama için). None → standart ses.
        language     : Dil kodu. Varsayılan "tr" (Türkçe).
        tone_profile : "enerjik" | "kurumsal" | "dramatik" | "samimi" | "lüks"
        speed        : Konuşma hızı (1.0 = normal). None → profil hızı.

        Dönüş
        ------
        str : Oluşturulan ses dosyasının yolu.
        """
        self.load_model()  # Henüz yüklenmediyse yükle

        # Türkçe karakterlerin UTF-8 olarak düzgün işlenmesi
        text = self._normalize_turkish(text)

        # Ton profili hız ayarı
        profile = AD_TONE_PROFILES.get(tone_profile, AD_TONE_PROFILES["kurumsal"])
        _speed  = speed if speed is not None else profile.get("speed", 1.0)

        logger.info(
            "Ses sentezi başlıyor | Ton: %s | Hız: %.2f | Dil: %s | Karakter: %d",
            tone_profile, _speed, language, len(text)
        )

        try:
            if speaker_wav:
                # Ses klonlama modu – referans ses zorunlu
                self.model.tts_to_file(
                    text        = text,
                    file_path   = output_path,
                    speaker_wav = speaker_wav,
                    language    = language,
                    speed       = _speed,
                )
            else:
                # Standart Türkçe ses sentezi
                self.model.tts_to_file(
                    text      = text,
                    file_path = output_path,
                    language  = language,
                    speed     = _speed,
                )
        except Exception as exc:
            logger.error("TTS hatası: %s", exc)
            raise

        logger.info("Ses dosyası kaydedildi: %s", output_path)
        return output_path

    # ──────────────────────────────────────────
    # MEVCUT SESLERİ LİSTELE
    # ──────────────────────────────────────────
    def list_speakers(self) -> list:
        """
        Modeldeki hazır konuşmacı listesini döner.
        Klonlama modunda kullanılmaz.
        """
        self.load_model()
        try:
            return self.model.speakers or []
        except Exception:
            return []

    # ──────────────────────────────────────────
    # TÜRKÇE KARAKTER NORMALİZASYONU
    # ──────────────────────────────────────────
    @staticmethod
    def _normalize_turkish(text: str) -> str:
        """
        Türkçe metni TTS motoruna uygun hale getirir:
        - UTF-8 kodlamasını doğrular
        - Yaygın yazım hatalarını düzeltir
        - Noktalama temizler (isteğe bağlı)
        """
        # Python str nesnesi zaten unicode; encoding kontrolü
        if not isinstance(text, str):
            text = text.decode("utf-8", errors="replace")

        # Yaygın ASCII → Türkçe karakter dönüşümleri
        _replacements = {
            "i̇": "İ",   # baş harf dönüşüm sorunu
            " ' " : "'",  # akıllı tırnak
        }
        for src, dst in _replacements.items():
            text = text.replace(src, dst)

        return text.strip()
