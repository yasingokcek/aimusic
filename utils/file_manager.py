# -*- coding: utf-8 -*-
"""
file_manager.py – Dosya Yönetimi ve Temizleme
==============================================
Yükleme/indirme, geçici dosya temizliği ve
bulut depolama entegrasyonu için araçlar.
"""

import uuid
import shutil
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing  import Optional, List

sys_path_inserted = False
import sys
if not sys_path_inserted:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import UPLOADS_DIR, VOICES_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_FILE_SIZE_MB  = 100


class FileManager:
    """
    Proje dosya yaşam döngüsü yöneticisi.

    Sorumluluklar:
    - Güvenli dosya yükleme (uzantı + boyut kontrolü)
    - Benzersiz dosya adı üretimi
    - Geçici dosya temizliği (TTL tabanlı)
    - Dosya listesi ve metadata sorgulama
    """

    def __init__(
        self,
        uploads_dir : Optional[Path] = None,
        outputs_dir : Optional[Path] = None,
        voices_dir  : Optional[Path] = None,
    ):
        self.uploads_dir = uploads_dir or UPLOADS_DIR
        self.outputs_dir = outputs_dir or OUTPUTS_DIR
        self.voices_dir  = voices_dir  or VOICES_DIR

        for d in [self.uploads_dir, self.outputs_dir, self.voices_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # GÜVENLİ DOSYA KAYDETME
    # ──────────────────────────────────────────
    def save_upload(
        self,
        file_content : bytes,
        original_name: str,
        target_dir   : Optional[str] = None,
    ) -> str:
        """
        Yüklenen dosyayı güvenli biçimde kaydeder.

        Kontroller:
        - Dosya uzantısı (yalnızca ses formatları)
        - Dosya boyutu (<= MAX_FILE_SIZE_MB)
        - Dosya adını sanitize et (path traversal önleme)

        Parametreler
        ------------
        file_content  : Ham dosya baytları.
        original_name : Orijinal dosya adı.
        target_dir    : Hedef klasör yolu (None → uploads_dir).

        Dönüş
        ------
        str : Kaydedilen dosyanın tam yolu.
        """
        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXT:
            raise ValueError(
                f"İzin verilmeyen dosya formatı: {ext}. "
                f"İzin verilenler: {ALLOWED_AUDIO_EXT}"
            )

        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Dosya çok büyük: {size_mb:.1f} MB. "
                f"Maksimum: {MAX_FILE_SIZE_MB} MB"
            )

        dest = Path(target_dir) if target_dir else self.uploads_dir
        dest.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex[:8]}_{self._sanitize_filename(original_name)}"
        file_path   = dest / unique_name

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info("Dosya kaydedildi: %s (%.2f MB)", file_path, size_mb)
        return str(file_path)

    # ──────────────────────────────────────────
    # ÇIKTI DOSYASI YOLU ÜRET
    # ──────────────────────────────────────────
    def new_output_path(self, prefix: str = "output", ext: str = "wav") -> str:
        """
        Benzersiz çıktı dosya yolu üretir.
        Örn: outputs/output_20240115_a3b4c5d6.wav
        """
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid       = uuid.uuid4().hex[:8]
        filename  = f"{prefix}_{ts}_{uid}.{ext}"
        return str(self.outputs_dir / filename)

    # ──────────────────────────────────────────
    # ÇIKTI LİSTESİ
    # ──────────────────────────────────────────
    def list_outputs(self, ext: str = "wav") -> List[dict]:
        """
        Üretilmiş çıktı dosyalarını listeler.

        Dönüş
        ------
        list[dict] : {"name", "path", "size_mb", "created_at"}
        """
        results = []
        for f in sorted(self.outputs_dir.glob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True):
            stat = f.stat()
            results.append({
                "name"      : f.name,
                "path"      : str(f),
                "size_mb"   : round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        return results

    # ──────────────────────────────────────────
    # GEÇİCİ DOSYA TEMİZLİĞİ
    # ──────────────────────────────────────────
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Belirtilen saatten eski çıktı dosyalarını siler.

        Parametreler
        ------------
        max_age_hours : Dosyaların TTL süresi saat cinsinden.

        Dönüş
        ------
        int : Silinen dosya sayısı.
        """
        cutoff   = datetime.now() - timedelta(hours=max_age_hours)
        count    = 0
        for f in self.outputs_dir.rglob("*"):
            if not f.is_file():
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink()
                count += 1
        logger.info("Eski dosyalar temizlendi: %d dosya silindi.", count)
        return count

    # ──────────────────────────────────────────
    # YARDIMCI METODLAR
    # ──────────────────────────────────────────
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Dosya adından zararlı karakterleri temizler.
        Path traversal saldırılarını önler.
        """
        import re
        # Sadece harf, rakam, nokta, tire, alt çizgi
        safe = re.sub(r"[^\w.\-]", "_", Path(name).name, flags=re.UNICODE)
        # Çift nokta kontrolü (../ saldırısı)
        safe = safe.replace("..", "_")
        return safe[:100]   # Max 100 karakter

    @staticmethod
    def get_checksum(file_path: str) -> str:
        """SHA-256 checksum döner (dosya bütünlüğü doğrulaması için)."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
