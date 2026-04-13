# -*- coding: utf-8 -*-
"""
audio_processor.py – Ses İşleme Yardımcıları
==============================================
Pydub ve FFmpeg tabanlı ses dönüştürme, normalleştirme
ve format işleme araçları.

Bağımlılıklar:
    pip install pydub librosa soundfile
    Sistem: ffmpeg (apt install ffmpeg)
"""

import logging
from pathlib import Path
from typing  import Optional, Tuple

import numpy  as np
import librosa             # type: ignore
import soundfile as sf     # type: ignore
from pydub import AudioSegment   # type: ignore

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Ses dosyaları üzerinde temel işlemler gerçekleştirir.

    Desteklenen formatlar: .wav, .mp3, .ogg, .flac, .m4a
    Çıkış formatı: .wav (varsayılan) veya .mp3
    """

    # ──────────────────────────────────────────
    # FORMAT DÖNÜŞTÜRME
    # ──────────────────────────────────────────
    @staticmethod
    def convert(
        src_path    : str,
        dst_path    : str,
        sample_rate : int  = 44100,
        channels    : int  = 2,
        bit_depth   : int  = 16,
    ) -> str:
        """
        Ses dosyasını istenen parametrelere dönüştürür.

        Parametreler
        ------------
        src_path    : Kaynak dosya yolu.
        dst_path    : Hedef dosya yolu (format uzantıdan anlaşılır).
        sample_rate : Hz cinsinden örnekleme hızı.
        channels    : 1 = mono, 2 = stereo.
        bit_depth   : 16 veya 24.

        Dönüş
        ------
        str : Hedef dosya yolu.
        """
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(channels)
        audio = audio.set_sample_width(bit_depth // 8)

        ext = Path(dst_path).suffix.lower().lstrip(".")
        fmt = "mp3" if ext == "mp3" else "wav"
        audio.export(dst_path, format=fmt)
        logger.debug("Dönüştürüldü: %s → %s", src_path, dst_path)
        return dst_path

    # ──────────────────────────────────────────
    # SES NORMALIZASYON (LUFS tabanlı)
    # ──────────────────────────────────────────
    @staticmethod
    def normalize(
        audio_path  : str,
        output_path : str,
        target_dbfs : float = -3.0,
    ) -> str:
        """
        Ses dosyasının peak seviyesini hedef dBFS'e normalize eder.

        Parametreler
        ------------
        audio_path  : Kaynak dosya.
        output_path : Normalize edilmiş çıktı.
        target_dbfs : Hedef maksimum dBFS (-3.0 yayın standardı).
        """
        audio = AudioSegment.from_file(audio_path)
        delta = target_dbfs - audio.dBFS
        normalized = audio.apply_gain(delta)
        normalized.export(output_path, format="wav")
        logger.debug("Normalize edildi → %.1f dBFS: %s", target_dbfs, output_path)
        return output_path

    # ──────────────────────────────────────────
    # SES KIRMAK (TRIM)
    # ──────────────────────────────────────────
    @staticmethod
    def trim_silence(
        audio_path    : str,
        output_path   : str,
        silence_thresh: int = -50,   # dBFS
        chunk_size    : int = 10,    # ms
    ) -> str:
        """
        Ses dosyasının başındaki ve sonundaki sessizliği kaldırır.
        """
        from pydub.silence import detect_leading_silence  # type: ignore

        audio      = AudioSegment.from_file(audio_path)
        start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh, chunk_size=chunk_size)
        end_trim   = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
        duration   = len(audio)
        trimmed    = audio[start_trim : duration - end_trim]
        trimmed.export(output_path, format="wav")
        logger.debug("Sessizlik kırpıldı: %dms → %dms", duration, len(trimmed))
        return output_path

    # ──────────────────────────────────────────
    # SES PARÇALAMA
    # ──────────────────────────────────────────
    @staticmethod
    def split_by_duration(
        audio_path   : str,
        output_dir   : str,
        chunk_sec    : int = 10,
    ) -> list:
        """
        Sesi belirli süre parçalara böler (eğitim veri hazırlama).

        Dönüş
        ------
        list[str] : Parça dosya yolları listesi.
        """
        audio     = AudioSegment.from_file(audio_path)
        chunk_ms  = chunk_sec * 1000
        out_dir   = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths     = []

        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start : start + chunk_ms]
            if len(chunk) < 1000:   # 1 saniyeden kısa parçaları atla
                continue
            chunk_path = str(out_dir / f"chunk_{i:04d}.wav")
            chunk.export(chunk_path, format="wav")
            paths.append(chunk_path)

        logger.info("%d parça oluşturuldu: %s", len(paths), output_dir)
        return paths

    # ──────────────────────────────────────────
    # SES BİLGİSİ
    # ──────────────────────────────────────────
    @staticmethod
    def get_info(audio_path: str) -> dict:
        """
        Ses dosyasının temel bilgilerini döner.

        Dönüş
        ------
        dict : {"duration_sec", "sample_rate", "channels", "format", "size_mb"}
        """
        audio   = AudioSegment.from_file(audio_path)
        size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        return {
            "duration_sec" : len(audio) / 1000.0,
            "sample_rate"  : audio.frame_rate,
            "channels"     : audio.channels,
            "format"       : Path(audio_path).suffix.lower(),
            "size_mb"      : round(size_mb, 2),
        }

    # ──────────────────────────────────────────
    # SES GÖRSELLEŞTIRME VERİSİ (waveform)
    # ──────────────────────────────────────────
    @staticmethod
    def get_waveform(audio_path: str, num_points: int = 500) -> Tuple[np.ndarray, float]:
        """
        Waveform görselleştirme için örneklenmiş ses genlikleri döner.

        Dönüş
        ------
        (np.ndarray, float) : (genlik dizisi, süre saniye)
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration  = len(audio) / sr
        step      = max(1, len(audio) // num_points)
        waveform  = audio[::step][:num_points]
        return waveform, duration
