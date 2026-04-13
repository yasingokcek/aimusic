# -*- coding: utf-8 -*-
"""
music_generator.py – AI Müzik Üretim Modülü
=============================================
Meta'nın MusicGen modeli kullanılarak metin tanımından
arka plan müziği üretir.

Bağımlılıklar:
    pip install transformers scipy torch torchaudio audiocraft
"""

import sys
import logging
from pathlib import Path
from typing  import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MUSIC_CONFIG, AD_TONE_PROFILES, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    Metinden müzik üreten AI motoru.

    Özellikler:
    - MusicGen (facebook/musicgen-*) model desteği
    - Reklam tonu profiline göre otomatik prompt oluşturma
    - Özel süre ayarı (saniye cinsinden)
    - GPU/CPU otomatik seçimi
    """

    def __init__(self, model_name: Optional[str] = None, use_gpu: Optional[bool] = None):
        """
        Parametreler
        ------------
        model_name : str, isteğe bağlı
            HuggingFace model kimliği.
            "facebook/musicgen-small"  – ~300MB, hızlı
            "facebook/musicgen-medium" – ~1.5GB, dengeli
            "facebook/musicgen-large"  – ~3.3GB, en iyi kalite
        use_gpu : bool, isteğe bağlı
            None ise config.py ayarı kullanılır.
        """
        _gpu         = use_gpu if use_gpu is not None else MUSIC_CONFIG.get("use_gpu", True)
        self.device  = "cuda" if (_gpu and torch.cuda.is_available()) else "cpu"
        self.model_name = model_name or MUSIC_CONFIG["model"]
        self.model      = None
        self.processor  = None
        self._loaded    = False

        logger.info("MusicGenerator başlatıldı | Model: %s | Cihaz: %s",
                    self.model_name, self.device)

    # ──────────────────────────────────────────
    # MODEL YÜKLEME
    # ──────────────────────────────────────────
    def load_model(self) -> None:
        """
        MusicGen modelini ve işlemcisini belleğe yükler.
        İlk çağrıda model HuggingFace'den indirilir.
        """
        if self._loaded:
            return

        try:
            from audiocraft.models import MusicGen  # type: ignore
            logger.info("AudioCraft (MusicGen) yükleniyor: %s", self.model_name)

            # Model adını sadece kısa adıyla iste (small/medium/large)
            short_name   = self.model_name.split("/")[-1].replace("musicgen-", "")
            self.model   = MusicGen.get_pretrained(short_name)
            self._loaded = True
            logger.info("MusicGen modeli başarıyla yüklendi.")

        except ImportError:
            # AudioCraft kurulu değilse transformers ile dene
            logger.warning("audiocraft bulunamadı, transformers kullanılıyor.")
            self._load_with_transformers()

    def _load_with_transformers(self) -> None:
        """
        AudioCraft yoksa HuggingFace transformers pipeline ile yükler.
        """
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ne audiocraft ne de transformers kurulu. "
                "Lütfen çalıştırın: pip install audiocraft transformers"
            ) from exc

        logger.info("Transformers ile MusicGen yükleniyor: %s", self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model     = MusicgenForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        self._loaded   = True

    # ──────────────────────────────────────────
    # MÜZİK ÜRETME (AUDIOCRAFT)
    # ──────────────────────────────────────────
    def generate(
        self,
        prompt      : str,
        output_path : str,
        duration    : int           = 30,
        tone_profile: Optional[str] = None,
    ) -> str:
        """
        Metin tanımından arka plan müziği üretir.

        Parametreler
        ------------
        prompt       : Müzik tanımı (İngilizce en iyi sonucu verir).
                       Örn: "upbeat electronic music for tech ad, 30 seconds"
        output_path  : Çıktı .wav dosya yolu.
        duration     : Saniye cinsinden müzik uzunluğu.
        tone_profile : Reklam tonu profili (otomatik prompt zenginleştirir).

        Dönüş
        ------
        str : Oluşturulan müzik dosyasının yolu.
        """
        self.load_model()

        # Ton profili ile prompt zenginleştirme
        if tone_profile and tone_profile in AD_TONE_PROFILES:
            profile_prompt = AD_TONE_PROFILES[tone_profile].get("music_prompt", "")
            prompt = f"{prompt}, {profile_prompt}" if prompt else profile_prompt

        logger.info(
            "Müzik üretimi başlıyor | Prompt: '%s' | Süre: %ds", prompt, duration
        )

        try:
            from audiocraft.models import MusicGen  # type: ignore
            return self._generate_audiocraft(prompt, output_path, duration)
        except ImportError:
            return self._generate_transformers(prompt, output_path, duration)

    def _generate_audiocraft(
        self, prompt: str, output_path: str, duration: int
    ) -> str:
        """AudioCraft API ile müzik üretir."""
        import scipy.io.wavfile as wav  # type: ignore

        self.model.set_generation_params(duration=duration)
        outputs = self.model.generate([prompt])           # (1, channels, samples)
        audio   = outputs[0, 0].cpu().numpy()             # mono → 1D array

        sr = self.model.sample_rate
        wav.write(output_path, sr, audio.astype(np.float32))
        logger.info("Müzik kaydedildi (AudioCraft): %s", output_path)
        return output_path

    def _generate_transformers(
        self, prompt: str, output_path: str, duration: int
    ) -> str:
        """Transformers pipeline ile müzik üretir (fallback)."""
        import scipy.io.wavfile as wav  # type: ignore

        inputs = self.processor(
            text            = [prompt],
            padding         = True,
            return_tensors  = "pt",
        ).to(self.device)

        # Token sayısı → süre dönüşümü (yaklaşık 50 token/sn)
        max_new_tokens = duration * 50

        with torch.no_grad():
            audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        sr    = self.model.config.audio_encoder.sampling_rate
        audio = audio_values[0, 0].cpu().numpy()
        wav.write(output_path, sr, audio.astype(np.float32))
        logger.info("Müzik kaydedildi (Transformers): %s", output_path)
        return output_path

    # ──────────────────────────────────────────
    # HAZIR PROMPT ŞABLONLARı
    # ──────────────────────────────────────────
    @staticmethod
    def build_prompt(
        ad_type   : str = "genel",
        duration  : int = 30,
        extra     : str = "",
    ) -> str:
        """
        Reklam türüne göre hazır müzik prompt'u oluşturur.

        Parametreler
        ------------
        ad_type  : "teknoloji" | "moda" | "gida" | "otomotiv" | "saglik" | "genel"
        duration : Saniye cinsinden süre.
        extra    : Kullanıcının ek tanımı.
        """
        templates = {
            "teknoloji" : "modern minimalist electronic music, futuristic synths, clean beats",
            "moda"      : "elegant fashion background music, trendy pop, light percussion",
            "gida"      : "warm acoustic background music, appetizing, light and cheerful",
            "otomotiv"  : "powerful cinematic music, strong bass, driving rhythm",
            "saglik"    : "calm wellness music, soft piano, nature sounds, healing",
            "genel"     : "professional background music, neutral, clean production",
        }
        base    = templates.get(ad_type, templates["genel"])
        prompt  = f"{base}, {duration} seconds"
        if extra:
            prompt += f", {extra}"
        return prompt
