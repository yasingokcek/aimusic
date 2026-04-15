# -*- coding: utf-8 -*-
"""
Reklam Spotu Üreticisi
Türkçe metin → Gemini TTS → otomatik müzik mix → reklam spotu
"""

import io
import re
import time
import wave as wavemodule
import base64
import numpy as np
import scipy.io.wavfile as spwav
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
.header { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.header p  { color: #888; margin: .4rem 0 0; font-size: 1rem; }
.result-label {
    font-size: .75rem; font-weight: 600; color: #6C63FF;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: .5rem;
}
.tone-pill {
    display: inline-block;
    background: #6C63FF22; color: #6C63FF;
    border: 1px solid #6C63FF55;
    border-radius: 20px; padding: 2px 12px;
    font-size: .8rem; font-weight: 600;
}
.api-box {
    background: #1a1a2e; border: 1px solid #6C63FF44;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── SABITLER ─────────────────────────────────────────────────────────────────
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

TONE_CHORDS = {
    "enerjik" : [261, 329, 392, 523],
    "kurumsal": [220, 277, 330, 440],
    "dramatik": [196, 247, 294, 392],
    "samimi"  : [264, 330, 396, 528],
    "lüks"    : [277, 349, 415, 554],
}

GEMINI_VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]

# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────

def detect_tone(text: str) -> tuple[str, int]:
    tl = text.lower()
    scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in TONE_KEYWORDS.items()}
    best  = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    return best, int(scores[best] / total * 100)


def make_voice_gemini(text: str, api_key: str, voice: str = "Aoede") -> bytes:
    """Gemini TTS ile Türkçe ses üretir."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    prompt = (
        "Aşağıdaki Türkçe reklam metnini profesyonel, akıcı ve doğal bir "
        "seslendirme tonu ile oku. Sadece metni oku, ek yorum ekleme.\n\n"
        f"{text}"
    )

    response = genai.GenerativeModel("gemini-2.5-flash-preview-tts").generate_content(
        contents=prompt,
        generation_config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": voice}
                }
            },
        },
    )

    part      = response.candidates[0].content.parts[0]
    raw_audio = base64.b64decode(part.inline_data.data)
    mime      = part.inline_data.mime_type  # örn. "audio/pcm;rate=24000"

    if "wav" in mime.lower():
        return raw_audio

    # PCM → WAV
    rate_m = re.search(r"rate=(\d+)", mime)
    sr     = int(rate_m.group(1)) if rate_m else 24000
    buf    = io.BytesIO()
    with wavemodule.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(raw_audio)
    return buf.getvalue()


def make_music(tone: str, duration: int) -> bytes:
    """Tona göre arka plan müziği üretir."""
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
    spwav.write(buf, sr, (wave * 32767 * 0.7).astype(np.int16))
    return buf.getvalue()


def mix_audio(voice: bytes, music: bytes, duck_ratio: float = 0.25) -> bytes:
    """Ses + müzik birleştirir, ducking uygular."""
    def load(b):
        sr, d = spwav.read(io.BytesIO(b))
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
    buf   = io.BytesIO()
    spwav.write(buf, vsr, (mixed * 32767 * 0.9).astype(np.int16))
    return buf.getvalue()


def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


# ── SAYFA ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header">
  <h1>🎙️ Reklam Spotu Üreticisi</h1>
  <p>Türkçe metninizi yazın — Gemini TTS ile seslendir, müzikle mixle</p>
</div>
""", unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.markdown("""
<div class="api-box">
<b>🔑 Gemini API Anahtarı Gerekli</b><br><br>
Streamlit Cloud → <b>Manage app → Secrets</b> bölümüne şunu ekle:<br><br>
<code>GEMINI_API_KEY = "AIza..."</code>
</div>
""", unsafe_allow_html=True)
    st.stop()

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

if ad_text.strip():
    tone_auto, conf = detect_tone(ad_text)
    words = len(ad_text.split())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tespit edilen ton", TONE_LABELS[tone_auto])
    c2.metric("Tahmini süre", f"{round(words * 0.45, 1)}s")
    c3.metric("Kelime sayısı", words)

st.markdown("---")

with st.expander("⚙️ Gelişmiş ayarlar", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        tone_override = st.selectbox(
            "Müzik tonu",
            ["Otomatik"] + list(TONE_LABELS.values()),
        )
        gemini_voice = st.selectbox(
            "Gemini ses",
            GEMINI_VOICES,
            help="Aoede: kadın, Charon: erkek, Fenrir: derin erkek, Kore: genç kadın, Puck: nötr",
        )
    with col2:
        music_dur  = st.slider("Müzik süresi (sn)", 10, 60, 30, 5)
        duck_ratio = st.slider("Ducking oranı", 0.1, 0.6, 0.25, 0.05)

generate = st.button(
    "🎙️ Reklam Spotu Oluştur",
    type="primary",
    use_container_width=True,
    disabled=not ad_text.strip(),
)

if generate and ad_text.strip():

    tone = (
        next(k for k, v in TONE_LABELS.items() if v == tone_override)
        if tone_override != "Otomatik"
        else tone_auto
    )

    prog   = st.progress(0)
    status = st.empty()

    try:
        status.info("**1/3** — Gemini TTS ile ses üretiliyor…")
        voice_bytes = make_voice_gemini(ad_text, api_key, gemini_voice)
        prog.progress(40)

        status.info(f"**2/3** — Müzik üretiliyor ({TONE_LABELS[tone]}, {music_dur}s)…")
        music_bytes = make_music(tone, music_dur)
        prog.progress(75)

        status.info("**3/3** — Mix yapılıyor…")
        final_bytes = mix_audio(voice_bytes, music_bytes, duck_ratio)
        prog.progress(100)
        status.success("✅ Reklam spotu hazır!")

    except Exception as e:
        prog.empty()
        status.error(f"Hata: {e}")
        st.stop()

    st.markdown("---")
    st.markdown("### 🎧 Sonuçlar")

    col_v, col_m, col_f = st.columns(3)

    with col_v:
        st.markdown('<div class="result-label">Seslendirme (Gemini)</div>', unsafe_allow_html=True)
        st.audio(voice_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", voice_bytes, "seslendirme.wav", "audio/wav",
                           use_container_width=True)

    with col_m:
        st.markdown('<div class="result-label">Arka Plan Müziği</div>', unsafe_allow_html=True)
        st.audio(music_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", music_bytes, "muzik.wav", "audio/wav",
                           use_container_width=True)

    with col_f:
        st.markdown('<div class="result-label">Final Mix</div>', unsafe_allow_html=True)
        st.audio(final_bytes, format="audio/wav")
        st.download_button("⬇️ İndir", final_bytes, "reklam_spotu.wav", "audio/wav",
                           use_container_width=True, type="primary")

    st.markdown(
        f"Kullanılan ton: <span class='tone-pill'>{TONE_LABELS[tone]}</span> &nbsp;"
        f"Ses: <span class='tone-pill'>{gemini_voice}</span>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div style="text-align:center;opacity:.4;font-size:.75rem;margin-top:3rem">
Gemini 2.5 Flash TTS • Otomatik Müzik Mix
</div>
""", unsafe_allow_html=True)
