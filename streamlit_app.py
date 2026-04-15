# -*- coding: utf-8 -*-
"""
Reklam Spotu Üreticisi
Türkçe metin → ses sentezi → otomatik müzik mix → reklam spotu
"""

import io
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import streamlit as st

st.set_page_config(
    page_title="Reklam Spotu Üreticisi",
    page_icon="🎙️",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.header p  { color: #888; margin: .4rem 0 0; font-size: 1rem; }

.result-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: .75rem;
    font-weight: 600;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .5rem;
}
.tone-pill {
    display: inline-block;
    background: #6C63FF22;
    color: #6C63FF;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── SABIT VERİLER ─────────────────────────────────────────────────────────────
TONE_KEYWORDS = {
    "enerjik" : ["hızlı","dinamik","enerji","harika","heyecan","şimdi","bugün","fırsat","kampanya","indirim","acele"],
    "kurumsal": ["güvenilir","profesyonel","kalite","lider","deneyim","çözüm","hizmet","teknoloji","sistem","güç"],
    "dramatik": ["dönüşüm","devrim","tarihi","büyük","muhteşem","inanılmaz","özgür","kahraman","sınırsız","cesur"],
    "samimi"  : ["aile","sevgi","birlikte","paylaşım","mutluluk","sıcak","doğal","gerçek","güven","çocuk"],
    "lüks"    : ["premium","lüks","özel","seçkin","zarif","sofistike","şık","prestij","marka","ayrıcalık"],
}

TONE_LABELS = {
    "enerjik" : "⚡ Enerjik",
    "kurumsal": "🏢 Kurumsal",
    "dramatik": "🎭 Dramatik",
    "samimi"  : "💬 Samimi",
    "lüks"    : "💎 Lüks",
}

TONE_FREQ = {
    "enerjik" : 280,
    "kurumsal": 180,
    "dramatik": 130,
    "samimi"  : 220,
    "lüks"    : 160,
}

TONE_CHORDS = {
    "enerjik" : [261, 329, 392, 523],
    "kurumsal": [220, 277, 330, 440],
    "dramatik": [196, 247, 294, 392],
    "samimi"  : [264, 330, 396, 528],
    "lüks"    : [277, 349, 415, 554],
}

# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────

def detect_tone(text: str) -> tuple[str, int]:
    """Metinden ton tespiti. (ton, güven%) döndürür."""
    tl = text.lower()
    scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in TONE_KEYWORDS.items()}
    best  = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    conf  = int(scores[best] / total * 100) if total else 0
    return best, conf


def make_voice(text: str, tone: str, speed: float) -> bytes:
    """TTS simülasyonu — sine wave ile ses üretir."""
    sr       = 22050
    freq     = TONE_FREQ.get(tone, 200)
    duration = max(2.0, len(text.split()) * 0.45 / speed)
    t        = np.linspace(0, duration, int(sr * duration))

    wave  = np.sin(2 * np.pi * freq * t)
    wave += 0.3  * np.sin(2 * np.pi * freq * 2 * t)
    wave += 0.15 * np.sin(2 * np.pi * freq * 3 * t)

    fade = int(sr * 0.15)
    env  = np.ones_like(t)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    wave *= env * 0.6

    buf = io.BytesIO()
    wav.write(buf, sr, (wave * 32767).astype(np.int16))
    return buf.getvalue()


def make_music(tone: str, duration: int) -> bytes:
    """Tone göre arka plan müziği üretir."""
    sr    = 32000
    freqs = TONE_CHORDS.get(tone, TONE_CHORDS["kurumsal"])
    t     = np.linspace(0, duration, sr * duration)
    wave  = np.zeros_like(t)

    for i, f in enumerate(freqs):
        wave += (0.25 / (i + 1)) * np.sin(2 * np.pi * f * t + i * np.pi / 4)
        wave += 0.05 * np.sin(2 * np.pi * (f / 2) * t)

    fade = int(sr * 1.0)
    wave[:fade]  *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    buf = io.BytesIO()
    wav.write(buf, sr, (wave * 32767 * 0.7).astype(np.int16))
    return buf.getvalue()


def mix(voice: bytes, music: bytes, duck_ratio: float = 0.25) -> bytes:
    """Ses + müzik birleştirir, konuşma anında müziği kısar (ducking)."""
    def load(b):
        sr, d = wav.read(io.BytesIO(b))
        return sr, d.astype(np.float32) / 32767

    vsr, v = load(voice)
    _,   m = load(music)

    if len(m) < len(v):
        m = np.tile(m, (len(v) // len(m)) + 1)
    m = m[:len(v)]

    win = int(vsr * 0.05)
    for i in range(0, len(v) - win, win):
        if np.sqrt(np.mean(v[i:i+win] ** 2)) > 0.01:
            m[i:i+win] *= duck_ratio

    mixed = np.clip(v * 1.0 + m * 0.35, -1.0, 1.0)
    buf = io.BytesIO()
    wav.write(buf, vsr, (mixed * 32767 * 0.9).astype(np.int16))
    return buf.getvalue()


# ── SAYFA ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header">
  <h1>🎙️ Reklam Spotu Üreticisi</h1>
  <p>Türkçe metninizi yazın — ses sentezi ve müzik mixini otomatik yapalım</p>
</div>
""", unsafe_allow_html=True)

# Metin girişi
ad_text = st.text_area(
    "Reklam metni",
    placeholder=(
        "Örnek:\n"
        "Türkiye'nin lider teknoloji markası TechNova, yapay zeka çözümleriyle "
        "işletmenizi geleceğe taşıyor. Hemen keşfet."
    ),
    height=160,
    label_visibility="collapsed",
)

# Canlı ton tespiti
if ad_text.strip():
    detected_tone, conf = detect_tone(ad_text)
    words = len(ad_text.split())
    est   = round(words * 0.45, 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tespit edilen ton", TONE_LABELS[detected_tone])
    c2.metric("Tahmini süre", f"{est}s")
    c3.metric("Kelime sayısı", words)

st.markdown("---")

# Ayarlar — gizlenmiş, opsiyonel
with st.expander("⚙️ Gelişmiş ayarlar", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        tone_override = st.selectbox(
            "Müzik tonu (boş = otomatik)",
            ["Otomatik"] + list(TONE_LABELS.values()),
        )
        speed = st.slider("Konuşma hızı", 0.7, 1.5, 1.0, 0.05)
    with col2:
        music_dur  = st.slider("Müzik süresi (sn)", 10, 60, 30, 5)
        duck_ratio = st.slider("Ducking oranı", 0.1, 0.6, 0.25, 0.05,
                               help="Konuşma sırasında müzik ne kadar kısılsın")

# Üret butonu
generate = st.button(
    "🎙️ Reklam Spotu Oluştur",
    type="primary",
    use_container_width=True,
    disabled=not ad_text.strip(),
)

# ── ÜRETİM ───────────────────────────────────────────────────────────────────

if generate and ad_text.strip():

    # Ton belirle
    if tone_override == "Otomatik":
        tone, _ = detect_tone(ad_text)
    else:
        # etiketten anahtara çevir
        tone = next(k for k, v in TONE_LABELS.items() if v == tone_override)

    prog   = st.progress(0)
    status = st.empty()

    status.info("**1/3** — Ses sentez ediliyor…")
    time.sleep(0.6)
    voice_bytes = make_voice(ad_text, tone, speed)
    prog.progress(33)

    status.info(f"**2/3** — Müzik üretiliyor ({TONE_LABELS[tone]}, {music_dur}s)…")
    time.sleep(0.8)
    music_bytes = make_music(tone, music_dur)
    prog.progress(66)

    status.info("**3/3** — Mix yapılıyor…")
    time.sleep(0.5)
    final_bytes = mix(voice_bytes, music_bytes, duck_ratio)
    prog.progress(100)
    status.success("✅ Reklam spotu hazır!")

    st.markdown("---")

    # ── Sonuçlar ─────────────────────────────────────────────
    st.markdown("### 🎧 Sonuçlar")

    col_v, col_m, col_f = st.columns(3)

    with col_v:
        st.markdown('<div class="result-label">Seslendirme</div>', unsafe_allow_html=True)
        st.audio(voice_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", voice_bytes,
                           f"seslendirme.wav", "audio/wav",
                           use_container_width=True)

    with col_m:
        st.markdown('<div class="result-label">Arka Plan Müziği</div>', unsafe_allow_html=True)
        st.audio(music_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", music_bytes,
                           f"muzik.wav", "audio/wav",
                           use_container_width=True)

    with col_f:
        st.markdown('<div class="result-label">Final Mix</div>', unsafe_allow_html=True)
        st.audio(final_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", final_bytes,
                           "reklam_spotu.wav", "audio/wav",
                           use_container_width=True, type="primary")

    # Ton bilgisi
    st.markdown(
        f"Kullanılan ton: <span class='tone-pill'>{TONE_LABELS[tone]}</span>",
        unsafe_allow_html=True,
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;opacity:.4;font-size:.75rem;margin-top:3rem">
Demo modu — Gerçek prodüksiyonda XTTS v2 + MusicGen çalışır
</div>
""", unsafe_allow_html=True)
