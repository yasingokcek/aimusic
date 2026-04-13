# -*- coding: utf-8 -*-
"""
turkish_utils.py – Türkçe Dil ve Kodlama Araçları
===================================================
Türkçe karakter sorunlarını çözen, metni ön işleyen
ve sayıları yazıya çeviren yardımcı araçlar.

Python 3, natively UTF-8 string'leri destekler,
ancak bazı kütüphaneler ek işlem gerektirebilir.
"""

import re
import locale
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# TÜRKÇE LOCALE AYARI
# ──────────────────────────────────────────────
def setup_turkish_locale() -> bool:
    """
    Sistem locale'ini Türkçe'ye ayarlar.
    Başarılı olursa True döner.
    """
    for loc in ("tr_TR.UTF-8", "tr_TR.utf8", "tr_TR", "Turkish"):
        try:
            locale.setlocale(locale.LC_ALL, loc)
            logger.info("Türkçe locale ayarlandı: %s", loc)
            return True
        except locale.Error:
            continue
    logger.warning("Türkçe locale ayarlanamadı, varsayılan kullanılıyor.")
    return False


class TurkishUtils:
    """
    Türkçe metin işleme yardımcıları.

    - UTF-8 / ASCII dönüşümleri
    - Büyük/küçük harf (Türkçe kuralları)
    - Sayıdan yazıya çevirme
    - Metin temizleme ve normalleştirme
    - Kısaltma açma (TTS için)
    """

    # ──────────────────────────────────────────
    # TÜRKÇE KARAKTER HARİTALARI
    # ──────────────────────────────────────────
    _TO_LOWER: Dict[str, str] = {
        "I"  : "ı",   # Türkçe büyük I → küçük ı (noktalı değil)
        "İ"  : "i",   # Türkçe büyük İ → küçük i (noktalı)
        "Ç"  : "ç",
        "Ğ"  : "ğ",
        "Ö"  : "ö",
        "Ş"  : "ş",
        "Ü"  : "ü",
    }

    _TO_UPPER: Dict[str, str] = {
        "ı"  : "I",   # Türkçe küçük ı → büyük I (noktasız)
        "i"  : "İ",   # Türkçe küçük i → büyük İ (noktalı)
        "ç"  : "Ç",
        "ğ"  : "Ğ",
        "ö"  : "Ö",
        "ş"  : "Ş",
        "ü"  : "Ü",
    }

    # TTS için yaygın kısaltmalar
    _ABBREVIATIONS: Dict[str, str] = {
        r"\bTL\b"   : "Türk Lirası",
        r"\bKDV\b"  : "katma değer vergisi",
        r"\bABD\b"  : "Amerika Birleşik Devletleri",
        r"\bAB\b"   : "Avrupa Birliği",
        r"\bSn\."   : "Sayın",
        r"\bDr\."   : "Doktor",
        r"\bProf\." : "Profesör",
        r"\bvb\."   : "ve benzeri",
        r"\bvs\."   : "ve saire",
        r"\bvd\."   : "ve diğerleri",
        r"\bHz\b"   : "hertz",
        r"\bGB\b"   : "gigabayt",
        r"\bMB\b"   : "megabayt",
    }

    # ──────────────────────────────────────────
    # BÜYÜK / KÜÇÜK HARF (TÜRKÇE KURALLARI)
    # ──────────────────────────────────────────
    @classmethod
    def to_lower(cls, text: str) -> str:
        """Türkçe büyük/küçük harf kurallarına göre küçültür."""
        for upper, lower in cls._TO_LOWER.items():
            text = text.replace(upper, lower)
        return text.lower()

    @classmethod
    def to_upper(cls, text: str) -> str:
        """Türkçe büyük/küçük harf kurallarına göre büyütür."""
        for lower, upper in cls._TO_UPPER.items():
            text = text.replace(lower, upper)
        return text.upper()

    # ──────────────────────────────────────────
    # SAYIDAN YAZIYA ÇEVİRME (TTS için)
    # ──────────────────────────────────────────
    @staticmethod
    def number_to_words(n: int) -> str:
        """
        Tam sayıyı Türkçe yazıya çevirir (0–9999).
        TTS motorunun "123" yerine "yüz yirmi üç" okuması için.

        Parametreler
        ------------
        n : 0 ile 9999 arasında tam sayı.

        Dönüş
        ------
        str : Türkçe yazılı sayı.
        """
        ones = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
        tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]

        if n < 0:
            return "eksi " + TurkishUtils.number_to_words(-n)
        if n == 0:
            return "sıfır"
        if n < 10:
            return ones[n]
        if n < 100:
            t, o = divmod(n, 10)
            return tens[t] + (" " + ones[o] if o else "")
        if n < 1000:
            h, rem = divmod(n, 100)
            prefix = "yüz" if h == 1 else ones[h] + " yüz"
            return prefix + (" " + TurkishUtils.number_to_words(rem) if rem else "")
        if n < 10000:
            th, rem = divmod(n, 1000)
            prefix = "bin" if th == 1 else ones[th] + " bin"
            return prefix + (" " + TurkishUtils.number_to_words(rem) if rem else "")

        return str(n)   # Büyük sayılar için fallback

    # ──────────────────────────────────────────
    # METİN NORMALİZASYONU
    # ──────────────────────────────────────────
    @classmethod
    def normalize_for_tts(cls, text: str) -> str:
        """
        Metni TTS motoruna uygun hale getirir:
        1. Kısaltmaları aç
        2. Sayıları yazıya çevir
        3. Özel karakterleri temizle
        4. Fazla boşlukları kaldır
        5. UTF-8 doğrula
        """
        if not isinstance(text, str):
            text = text.decode("utf-8", errors="replace")

        # ① Kısaltmaları aç
        for pattern, replacement in cls._ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.UNICODE)

        # ② Sayıları yazıya çevir (0-9999 arası)
        def replace_number(m):
            try:
                return cls.number_to_words(int(m.group()))
            except (ValueError, OverflowError):
                return m.group()

        text = re.sub(r"\b\d{1,4}\b", replace_number, text)

        # ③ URL ve e-posta adreslerini kaldır/sadeleştir
        text = re.sub(r"http[s]?://\S+", "web sitesi", text)
        text = re.sub(r"\S+@\S+\.\S+", "e-posta adresi", text)

        # ④ Çoklu noktalama → tek
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        # ⑤ Fazla boşluk
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ──────────────────────────────────────────
    # ASCII DÖNÜŞÜMÜ (gerektiğinde)
    # ──────────────────────────────────────────
    _TR_TO_ASCII: Dict[str, str] = {
        "ç": "c", "Ç": "C",
        "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S",
        "ü": "u", "Ü": "U",
    }

    @classmethod
    def to_ascii(cls, text: str) -> str:
        """
        Türkçe karakterleri ASCII karşılıklarıyla değiştirir.
        Yalnızca ASCII kabul eden sistemler (dosya adları vb.) için.
        """
        for tr_char, ascii_char in cls._TR_TO_ASCII.items():
            text = text.replace(tr_char, ascii_char)
        return text

    # ──────────────────────────────────────────
    # KARAKTER KONTROLÜ
    # ──────────────────────────────────────────
    @staticmethod
    def has_turkish_chars(text: str) -> bool:
        """Metinde Türkçe özel karakter var mı kontrol eder."""
        turkish = set("çÇğĞıİöÖşŞüÜ")
        return any(c in turkish for c in text)

    @staticmethod
    def ensure_utf8(text: str) -> str:
        """
        Metnin UTF-8 uyumlu olmasını garantiler.
        Bozuk karakterleri '?' ile değiştirir.
        """
        return text.encode("utf-8", errors="replace").decode("utf-8")
