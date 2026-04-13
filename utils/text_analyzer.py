# -*- coding: utf-8 -*-
"""
text_analyzer.py – Türkçe Metin Analizi ve Duygu Tanıma
=========================================================
Reklam metninin duygusal tonunu analiz ederek uygun
TTS tonu ve müzik stilini önerir.

Bağımlılıklar:
    pip install transformers torch
    (İsteğe bağlı gelişmiş analiz için)
"""

import re
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Türkçe reklam metni analiz aracı.

    Yetenekler:
    - Kelime bazlı duygu tonu tahmini (hızlı, bağımlılık gerektirmez)
    - Metin istatistikleri (karakter, kelime, tahmini süre)
    - Reklam kategorisi tespiti
    - Uygun TTS tonu ve müzik stili önerisi
    """

    # ──────────────────────────────────────────
    # TÜRKÇE DUYGU KELİME HAZİNESİ
    # ──────────────────────────────────────────
    _TONE_KEYWORDS: Dict[str, list] = {
        "enerjik": [
            "hızlı", "dinamik", "enerji", "güç", "süper", "harika", "mükemmel",
            "heyecan", "fırsatı", "kaçırma", "şimdi", "bugün", "yeni", "en iyi",
            "kazanç", "indirim", "fırsat", "sınırlı", "kampanya", "özel",
        ],
        "kurumsal": [
            "güvenilir", "profesyonel", "kalite", "lider", "deneyim", "çözüm",
            "iş", "hizmet", "kurumsal", "global", "uluslararası", "yönetim",
            "strateji", "performans", "verimlilik", "teknoloji", "sistem",
        ],
        "dramatik": [
            "dönüşüm", "değişim", "devrim", "tarihi", "efsane", "unutulmaz",
            "büyük", "muhteşem", "inanılmaz", "imkansız", "sınırsız", "özgür",
            "güçlü", "etkileyici", "cesur", "kahraman",
        ],
        "samimi": [
            "aile", "sevgi", "dostluk", "birlikte", "paylaşım", "mutluluk",
            "sıcak", "doğal", "gerçek", "dürüst", "samimi", "güven", "destek",
            "yardım", "anlayış", "beraber",
        ],
        "lüks": [
            "premium", "lüks", "özel", "seçkin", "zarif", "sofistike", "şık",
            "prestij", "kaliteli", "üstün", "rafine", "marka", "tasarım",
            "koleksiyon", "sınırlı", "istek",
        ],
    }

    # ──────────────────────────────────────────
    # TON ANALİZİ
    # ──────────────────────────────────────────
    def analyze_tone(self, text: str) -> Tuple[str, float, dict]:
        """
        Metindeki anahtar kelimelere göre reklam tonunu tahmin eder.

        Parametreler
        ------------
        text : Analiz edilecek Türkçe metin.

        Dönüş
        ------
        (tone, confidence, scores) :
            tone       : En yüksek puanlı ton.
            confidence : 0.0–1.0 arası güven skoru.
            scores     : {"enerjik": 0.8, "kurumsal": 0.2, ...}
        """
        text_lower = text.lower()
        words      = re.findall(r"\b\w+\b", text_lower, re.UNICODE)
        word_count = len(words)

        if word_count == 0:
            return "kurumsal", 0.0, {}

        scores: Dict[str, float] = {}
        for tone, keywords in self._TONE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            scores[tone] = matches / max(1, len(keywords))

        # En yüksek skoru al
        best_tone  = max(scores, key=scores.get)
        best_score = scores[best_tone]

        # Güven normalizasyonu
        total    = sum(scores.values()) or 1
        confidence = best_score / total

        logger.debug("Ton analizi: %s (%.2f)", best_tone, confidence)
        return best_tone, confidence, scores

    # ──────────────────────────────────────────
    # METİN İSTATİSTİKLERİ
    # ──────────────────────────────────────────
    @staticmethod
    def get_stats(text: str) -> dict:
        """
        Metin istatistiklerini ve tahmini seslendirme süresini döner.

        Dönüş
        ------
        dict : {
            "char_count"    : int,
            "word_count"    : int,
            "sentence_count": int,
            "estimated_sec" : float,   # Normal hızda ~130 kelime/dk
        }
        """
        text       = text.strip()
        char_count = len(text)
        words      = re.findall(r"\b\w+\b", text, re.UNICODE)
        word_count = len(words)
        sentences  = re.split(r"[.!?]+", text)
        sent_count = len([s for s in sentences if s.strip()])

        # Türkçe ortalama konuşma hızı: ~130–140 kelime/dakika
        estimated_sec = (word_count / 135) * 60

        return {
            "char_count"    : char_count,
            "word_count"    : word_count,
            "sentence_count": sent_count,
            "estimated_sec" : round(estimated_sec, 1),
        }

    # ──────────────────────────────────────────
    # REKLAM KATEGORİSİ TESPİTİ
    # ──────────────────────────────────────────
    def detect_category(self, text: str) -> str:
        """
        Metnin hangi sektör reklamı olduğunu tahmin eder.
        Müzik prompt önerisi için kullanılır.

        Dönüş: "teknoloji" | "moda" | "gida" | "otomotiv" | "saglik" | "genel"
        """
        _categories = {
            "teknoloji" : ["yazılım", "uygulama", "yapay zeka", "dijital", "teknoloji",
                           "bulut", "veri", "internet", "mobil", "akıllı"],
            "moda"      : ["moda", "giyim", "koleksiyon", "tasarım", "stil", "şık",
                           "tarz", "sezon", "trend", "aksesuar"],
            "gida"      : ["yemek", "içecek", "lezzet", "tarif", "taze", "doğal",
                           "organik", "besleyici", "gıda", "kahvaltı", "akşam"],
            "otomotiv"  : ["araba", "araç", "motor", "sürüş", "yol", "hız", "güç",
                           "yakıt", "oto", "elektrikli"],
            "saglik"    : ["sağlık", "ilaç", "vitamin", "egzersiz", "spor", "wellness",
                           "doğal", "organik", "terapi", "beslenme"],
        }
        text_lower = text.lower()
        best_cat, best_score = "genel", 0
        for cat, kws in _categories.items():
            score = sum(1 for kw in kws if kw in text_lower)
            if score > best_score:
                best_cat, best_score = cat, score
        return best_cat

    # ──────────────────────────────────────────
    # TAM ANALİZ
    # ──────────────────────────────────────────
    def full_analysis(self, text: str) -> dict:
        """
        Tüm analizleri birleştirerek kapsamlı rapor döner.

        Dönüş
        ------
        dict : {
            "tone"         : str,
            "confidence"   : float,
            "scores"       : dict,
            "category"     : str,
            "stats"        : dict,
            "music_prompt" : str,
        }
        """
        from config import AD_TONE_PROFILES
        from models.music_generator import MusicGenerator

        tone, conf, scores = self.analyze_tone(text)
        category           = self.detect_category(text)
        stats              = self.get_stats(text)

        # Müzik prompt önerisi
        profile_prompt = AD_TONE_PROFILES.get(tone, {}).get("music_prompt", "")
        music_prompt   = MusicGenerator.build_prompt(
            ad_type  = category,
            duration = int(stats["estimated_sec"]) + 5,  # +5s tampon
            extra    = profile_prompt,
        )

        return {
            "tone"        : tone,
            "confidence"  : round(conf, 3),
            "scores"      : scores,
            "category"    : category,
            "stats"       : stats,
            "music_prompt": music_prompt,
        }
