# -*- coding: utf-8 -*-
"""
mixer.py – Profesyonel Ses Karıştırma ve Mastering Modülü
==========================================================
Voiceover (seslendirme) ve arka plan müziğini birleştirir.
Ducking (oto ses kısma), fade in/out, mastering zincirleri
bu modülde uygulanır.

Bağımlılıklar:
    pip install pydub numpy
    Sistem: ffmpeg
"""

import logging
from pathlib import Path
from typing  import Optional

import numpy as np
from pydub import AudioSegment  # type: ignore
from pydub.effects import compress_dynamic_range  # type: ignore

sys.path_hack = None
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIX_CONFIG

logger = logging.getLogger(__name__)


class AudioMixer:
    """
    Seslendirme + Arka Plan Müziği Karıştırıcı.

    Özellikler:
    - Otomatik ducking (ses konuşmaya başladığında müziği kısar)
    - Fade in / fade out
    - Ses seviyesi normalizasyonu
    - Stereo master çıkışı
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Parametreler
        ------------
        config : dict, isteğe bağlı
            MIX_CONFIG anahtarlarını override eder.
        """
        self.cfg = {**MIX_CONFIG, **(config or {})}

    # ──────────────────────────────────────────
    # ANA MİX FONKSİYONU
    # ──────────────────────────────────────────
    def mix(
        self,
        voice_path  : str,
        music_path  : str,
        output_path : str,
        ducking     : bool  = True,
        duck_ratio  : float = 0.30,   # Konuşma sırasında müziğin kaç katına düşüleceği
    ) -> str:
        """
        Seslendirme ve arka plan müziğini birleştirir.

        Parametreler
        ------------
        voice_path  : TTS/seslendirme .wav dosyası.
        music_path  : Arka plan müziği .wav dosyası.
        output_path : Karışık final çıktı.
        ducking     : Otomatik ses kısma aktif mi?
        duck_ratio  : Konuşma sırasında müzik 1.0 oranından
                      bu orana düşer (0.30 = %30 ses).

        Dönüş
        ------
        str : Çıktı dosya yolu.
        """
        logger.info("Mix başlıyor | Ducking: %s", ducking)

        # ① Dosyaları yükle ve stereo'ya çevir
        voice = self._load_and_prepare(voice_path)
        music = self._load_and_prepare(music_path)

        # ② Ses seviyelerini ayarla
        voice = voice.apply_gain(self.cfg["voice_volume_db"])
        music = music.apply_gain(self.cfg["music_volume_db"])

        # ③ Müziği ses uzunluğuna göre ayarla (loop/kırp)
        music = self._fit_music_to_voice(music, voice)

        # ④ Fade in/out ekle
        music = music.fade_in(self.cfg["fade_in_ms"]).fade_out(self.cfg["fade_out_ms"])

        # ⑤ Ducking uygula
        if ducking:
            music = self._apply_ducking(voice, music, duck_ratio)

        # ⑥ Karıştır
        combined = voice.overlay(music)

        # ⑦ Master zinciri uygula
        combined = self._master(combined)

        # ⑧ Çıktıya kaydet
        fmt = Path(output_path).suffix.lstrip(".").lower() or "wav"
        combined.export(output_path, format=fmt)
        logger.info("Mix tamamlandı → %s (%.1fs)", output_path, len(combined) / 1000)
        return output_path

    # ──────────────────────────────────────────
    # DUCKING (OTO SES KISMA)
    # ──────────────────────────────────────────
    def _apply_ducking(
        self,
        voice       : AudioSegment,
        music       : AudioSegment,
        duck_ratio  : float,
    ) -> AudioSegment:
        """
        Konuşma (voice) aktifken arka plan müziğini duck_ratio oranında kısar.

        Algoritma:
        1. Voice üzerinde 50ms pencerelerle RMS enerji hesapla
        2. Sessizlik eşiğinin üstündeki pencereleri "aktif" işaretle
        3. Müzikteki o pencerelerde gain azalt
        4. Attach/release zamanları ile smooth geçiş sağla
        """
        WINDOW_MS     = 50
        SILENCE_THRESH = -40   # dBFS

        attack_ms  = self.cfg["duck_attack_ms"]
        release_ms = self.cfg["duck_release_ms"]
        duck_db    = 20 * np.log10(duck_ratio + 1e-10)   # 0.30 → ≈ -10.5 dB

        # Voice enerjisini pencere pencere hesapla
        voice_arr = np.array(voice.get_array_of_samples(), dtype=np.float32)
        sr        = voice.frame_rate
        win_n     = int(WINDOW_MS / 1000 * sr)
        n_wins    = max(1, len(voice_arr) // win_n)

        # Gain dizisi oluştur (her pencere için 1 değer)
        gain_schedule = []
        for i in range(n_wins):
            chunk = voice_arr[i * win_n : (i + 1) * win_n]
            rms   = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            # dBFS hesabı (16-bit normalize)
            dbfs  = 20 * np.log10(rms / 32768.0 + 1e-10)
            is_speech = dbfs > SILENCE_THRESH
            gain_schedule.append(duck_db if is_speech else 0.0)

        # Gain dizisini pencerelere uygula
        ducked_music = AudioSegment.empty()
        for i, gain in enumerate(gain_schedule):
            start_ms = i * WINDOW_MS
            end_ms   = min(start_ms + WINDOW_MS, len(music))
            if start_ms >= len(music):
                break
            segment = music[start_ms:end_ms]
            ducked_music += segment.apply_gain(gain)

        # Kalan müzik (voice'dan uzunsa)
        if len(music) > n_wins * WINDOW_MS:
            ducked_music += music[n_wins * WINDOW_MS:]

        logger.debug("Ducking uygulandı | Pencere: %dms | Gain: %.1fdB", WINDOW_MS, duck_db)
        return ducked_music

    # ──────────────────────────────────────────
    # MASTERING ZİNCİRİ
    # ──────────────────────────────────────────
    @staticmethod
    def _master(audio: AudioSegment) -> AudioSegment:
        """
        Basit mastering zinciri:
        1. Dinamik sıkıştırma (kompresör)
        2. Peak limiter (-0.5 dBFS)
        """
        # Kompresör
        try:
            audio = compress_dynamic_range(
                audio,
                threshold  = -20.0,
                ratio      = 4.0,
                attack     = 5.0,
                release    = 50.0,
            )
        except Exception as exc:
            logger.warning("Kompresör uygulanamadı: %s", exc)

        # Peak limiter (normalize to -0.5 dBFS)
        peak_db = audio.dBFS
        if peak_db > -0.5:
            audio = audio.apply_gain(-0.5 - peak_db)

        return audio

    # ──────────────────────────────────────────
    # YARDIMCI METODLAR
    # ──────────────────────────────────────────
    def _load_and_prepare(self, path: str) -> AudioSegment:
        """Ses dosyasını yükler ve standart formata çevirir."""
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(self.cfg["output_sample_rate"])
        audio = audio.set_channels(2)    # Stereo
        audio = audio.set_sample_width(2)   # 16-bit
        return audio

    @staticmethod
    def _fit_music_to_voice(music: AudioSegment, voice: AudioSegment) -> AudioSegment:
        """
        Müzik uzunluğunu sese eşitler.
        - Müzik kısaysa: döngüye alır.
        - Müzik uzunsa: sesi uzunluğuna kırpar.
        """
        voice_len = len(voice)
        music_len = len(music)

        if music_len < voice_len:
            # Müziği döngüye al (tekrarla)
            repeat_times = (voice_len // music_len) + 2
            music        = music * repeat_times

        return music[:voice_len]
