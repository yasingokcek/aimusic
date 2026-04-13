# -*- coding: utf-8 -*-
"""
voice_cloner.py – Ses Klonlama ve Fine-Tuning Modülü
=====================================================
Kullanıcıların yüklediği ses örneklerini kullanarak
XTTS v2 ile özel ses profili oluşturur.

Telif ve Etik Uyarısı:
    Bu modül yalnızca kullanıcının kendi sesi veya
    açık rıza alınmış sesler için kullanılmalıdır.
    Telif hakkı kontrolü _check_copyright() ile yapılır.

Bağımlılıklar:
    pip install TTS librosa soundfile numpy
"""

import sys
import uuid
import shutil
import hashlib
import logging
from pathlib import Path
from typing  import List, Tuple, Optional, Dict

import numpy  as np
import librosa             # type: ignore
import soundfile as sf     # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TTS_CONFIG, VOICES_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Minimum/maksimum kabul edilen ses uzunlukları (saniye)
MIN_DURATION_SEC = 10
MAX_DURATION_SEC = 600   # 10 dakika

# Ses örneği kalite eşikleri
MIN_SAMPLE_RATE  = 16000   # Hz
SNR_THRESHOLD_DB = 20      # Signal-to-Noise Ratio eşiği


class VoiceCloner:
    """
    Ses klonlama yöneticisi.

    Kullanım Akışı:
    1. validate_audio()   → ses kalitesini kontrol et
    2. register_voice()   → profil oluştur ve kaydet
    3. list_voices()      → kayıtlı profilleri listele
    4. get_voice_path()   → klonlama için dosya yolunu al
    5. delete_voice()     → profili sil
    """

    def __init__(self, voices_dir: Optional[Path] = None):
        self.voices_dir = voices_dir or VOICES_DIR
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._profile_db: Dict[str, dict] = {}   # Bellek içi profil deposu
        self._load_existing_profiles()

    # ──────────────────────────────────────────
    # SES DOĞRULAMA
    # ──────────────────────────────────────────
    def validate_audio(self, audio_path: str) -> Tuple[bool, str]:
        """
        Klonlama için yüklenen ses dosyasını doğrular.

        Kontroller:
        - Dosya formatı (.wav .mp3 .ogg .flac)
        - Ses uzunluğu (10 sn – 10 dk)
        - Örnekleme hızı (≥ 16 kHz)
        - Sinyal/Gürültü oranı (≥ 20 dB)

        Dönüş
        ------
        (bool, str) : (geçerli_mi, hata_mesajı_veya_OK)
        """
        path = Path(audio_path)

        # Format kontrolü
        allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        if path.suffix.lower() not in allowed:
            return False, f"Desteklenmeyen format: {path.suffix}. İzin verilenler: {allowed}"

        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as exc:
            return False, f"Ses dosyası okunamadı: {exc}"

        # Süre kontrolü
        duration = len(audio) / sr
        if duration < MIN_DURATION_SEC:
            return False, f"Ses çok kısa ({duration:.1f}s). En az {MIN_DURATION_SEC}s gerekli."
        if duration > MAX_DURATION_SEC:
            return False, f"Ses çok uzun ({duration:.1f}s). En fazla {MAX_DURATION_SEC}s kabul edilir."

        # Örnekleme hızı kontrolü
        if sr < MIN_SAMPLE_RATE:
            return False, f"Örnekleme hızı çok düşük ({sr} Hz). En az {MIN_SAMPLE_RATE} Hz gerekli."

        # SNR tahmini (basit yöntem: RMS tabanlı)
        snr = self._estimate_snr(audio)
        if snr < SNR_THRESHOLD_DB:
            return False, (
                f"Ses kalitesi düşük (SNR ≈ {snr:.1f} dB). "
                f"Arka plan gürültüsü çok fazla. En az {SNR_THRESHOLD_DB} dB gerekli."
            )

        return True, "OK"

    # ──────────────────────────────────────────
    # SES PROFİLİ KAYDETME
    # ──────────────────────────────────────────
    def register_voice(
        self,
        audio_path  : str,
        voice_name  : str,
        description : str = "",
        check_copyright: bool = True,
    ) -> dict:
        """
        Ses dosyasını klonlama profili olarak kaydeder.

        Parametreler
        ------------
        audio_path      : Kaynak ses dosyası yolu.
        voice_name      : Profil adı (Türkçe karakter desteklenir).
        description     : İsteğe bağlı açıklama.
        check_copyright : Basit telif kontrolü yapılsın mı?

        Dönüş
        ------
        dict : {"id", "name", "path", "duration", "sample_rate"}
        """
        # Önce doğrulama
        valid, msg = self.validate_audio(audio_path)
        if not valid:
            raise ValueError(f"Ses doğrulama başarısız: {msg}")

        # Telif kontrolü (hash tabanlı basit blacklist)
        if check_copyright:
            flagged, reason = self._check_copyright(audio_path)
            if flagged:
                raise PermissionError(f"Telif hakkı uyarısı: {reason}")

        # Benzersiz profil ID'si
        voice_id    = str(uuid.uuid4())[:8]
        safe_name   = self._sanitize_name(voice_name)
        dest_dir    = self.voices_dir / voice_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Ses dosyasını profil dizinine kopyala ve WAV'a dönüştür
        dest_path = dest_dir / "reference.wav"
        self._convert_to_wav(audio_path, str(dest_path))

        # Metadata
        audio, sr = librosa.load(str(dest_path), sr=None, mono=True)
        duration  = len(audio) / sr

        profile = {
            "id"          : voice_id,
            "name"        : voice_name,
            "safe_name"   : safe_name,
            "path"        : str(dest_path),
            "duration"    : round(duration, 2),
            "sample_rate" : sr,
            "description" : description,
        }

        # Metadata'yı kaydet
        self._profile_db[voice_id] = profile
        self._save_profile_meta(dest_dir, profile)

        logger.info("Ses profili kaydedildi: %s (ID: %s)", voice_name, voice_id)
        return profile

    # ──────────────────────────────────────────
    # PROFİL YÖNETİMİ
    # ──────────────────────────────────────────
    def list_voices(self) -> List[dict]:
        """Kayıtlı tüm ses profillerini döner."""
        return list(self._profile_db.values())

    def get_voice_path(self, voice_id: str) -> str:
        """
        ID'ye göre referans ses dosyası yolunu döner.
        VoiceEngine.synthesize() speaker_wav parametresinde kullanılır.
        """
        profile = self._profile_db.get(voice_id)
        if not profile:
            raise KeyError(f"Ses profili bulunamadı: {voice_id}")
        return profile["path"]

    def delete_voice(self, voice_id: str) -> bool:
        """Ses profilini ve tüm dosyalarını siler."""
        profile = self._profile_db.pop(voice_id, None)
        if not profile:
            return False
        dest_dir = self.voices_dir / voice_id
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        logger.info("Ses profili silindi: %s", voice_id)
        return True

    # ──────────────────────────────────────────
    # YARDIMCI METODLAR (ÖZEL)
    # ──────────────────────────────────────────
    def _load_existing_profiles(self) -> None:
        """Başlangıçta disk üzerindeki mevcut profilleri belleğe yükler."""
        import json
        for meta_file in self.voices_dir.rglob("meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                self._profile_db[profile["id"]] = profile
            except Exception as exc:
                logger.warning("Profil yüklenemedi (%s): %s", meta_file, exc)

    def _save_profile_meta(self, dest_dir: Path, profile: dict) -> None:
        """Profil metadata'sını JSON olarak kaydeder."""
        import json
        meta_path = dest_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _convert_to_wav(src: str, dst: str, target_sr: int = 22050) -> None:
        """Ses dosyasını 22050 Hz mono WAV formatına dönüştürür."""
        audio, _ = librosa.load(src, sr=target_sr, mono=True)
        sf.write(dst, audio, target_sr, subtype="PCM_16")

    @staticmethod
    def _estimate_snr(audio: np.ndarray) -> float:
        """
        RMS tabanlı basit SNR tahmini.
        Yüksek değer = düşük gürültü.
        """
        signal_rms = np.sqrt(np.mean(audio ** 2))
        # Gürültüyü alt %10'luk kısımdan tahmin et
        noise_floor = np.percentile(np.abs(audio), 10) + 1e-10
        snr         = 20 * np.log10(signal_rms / noise_floor)
        return float(snr)

    @staticmethod
    def _check_copyright(audio_path: str) -> Tuple[bool, str]:
        """
        Basit hash tabanlı telif kontrol mekanizması.
        Gerçek uygulamada ACRCloud veya AudD API entegrasyonu yapılabilir.

        Dönüş: (işaretlendi_mi, neden)
        """
        # Bilinen telif ihlali hash listesi (örnek – gerçekte API'den çekilir)
        KNOWN_HASHES: set = set()   # Boş küme = kontrol devre dışı

        try:
            with open(audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read(65536)).hexdigest()   # İlk 64KB

            if file_hash in KNOWN_HASHES:
                return True, "Bu ses dosyası telif hakkıyla korunan içerik listesinde bulunuyor."
        except Exception:
            pass

        return False, ""

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Dosya sisteminde güvenli kullanım için profil adını temizler.
        Türkçe karakterleri korur.
        """
        import re
        # Yalnızca harf, rakam, Türkçe karakter, boşluk ve tire
        safe = re.sub(r"[^\w\sçÇğĞıİöÖşŞüÜ-]", "", name, flags=re.UNICODE)
        return safe.strip()[:50]
