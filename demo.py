# -*- coding: utf-8 -*-
"""
demo.py – Canlı Demo (ML modeli gerektirmez)
=============================================
Gerçek TTS/MusicGen olmadan tüm UI akışını gösterir.
Mock ses üretimi: numpy ile sentezlenmiş sine wave.

Çalıştır:
    streamlit run demo.py --server.port 8501
"""

import io
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import streamlit as st

# ──────────────────────────────────────────────
# SAYFA YAPISI
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Müzik & Ses Prodüksiyon Stüdyosu — DEMO",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# TEMA / CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2rem; margin: 0; }
.hero p  { opacity: .85; margin: .4rem 0 0; font-size: 1rem; }

.badge {
    display: inline-block;
    background: rgba(255,255,255,.2);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: .8rem;
    margin: .5rem .2rem 0;
}

.step-box {
    background: #1a1a2e;
    border: 1px solid #2e2e5e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
}
.step-num {
    background: #6C63FF;
    color: white;
    border-radius: 50%;
    width: 24px; height: 24px;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 700;
    margin-right: .5rem;
}

.metric-card {
    background: #16213e;
    border-radius: 10px;
    padding: .8rem 1rem;
    text-align: center;
}
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #6C63FF; }
.metric-card .label { font-size: .75rem; opacity: .7; }

.tone-tag {
    display: inline-block;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: .8rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MOCK SES ÜRETİCİLERİ
# ──────────────────────────────────────────────

def make_voice_wav(text: str, tone: str, speed: float = 1.0) -> bytes:
    """
    Gerçek TTS yerine tonlama frekansına göre sine wave üretir.
    Prodüksiyonda bu fonksiyon VoiceEngine.synthesize() ile değiştirilir.
    """
    TONE_FREQ = {
        "enerjik" : 280,
        "kurumsal": 180,
        "dramatik": 130,
        "samimi"  : 220,
        "lüks"    : 160,
    }
    sr       = 22050
    freq     = TONE_FREQ.get(tone, 200)
    # Metin uzunluğuna göre süre (yaklaşık)
    duration = max(2.0, len(text.split()) * 0.45 / speed)
    t        = np.linspace(0, duration, int(sr * duration))

    # Doğal konuşma eğrisi simülasyonu (ADSR zarfı)
    wave   = np.sin(2 * np.pi * freq * t)
    # Harmonikler ekle (daha zengin ses)
    wave  += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    wave  += 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    # Doğal amplitüd zarfı
    env    = np.ones_like(t)
    fade   = int(sr * 0.15)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    wave  *= env * 0.6

    pcm = (wave * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, sr, pcm)
    return buf.getvalue()


def make_music_wav(prompt: str, duration: int, tone: str) -> bytes:
    """
    Gerçek MusicGen yerine ton profiline göre çok katmanlı müzik simüle eder.
    """
    TONE_CHORDS = {
        "enerjik" : [261, 329, 392, 523],   # C major arpeggio
        "kurumsal": [220, 277, 330, 440],   # A minor
        "dramatik": [196, 247, 294, 392],   # G minor
        "samimi"  : [264, 330, 396, 528],   # C major
        "lüks"    : [277, 349, 415, 554],   # C# jazz
    }
    sr     = 32000
    freqs  = TONE_CHORDS.get(tone, TONE_CHORDS["kurumsal"])
    t      = np.linspace(0, duration, sr * duration)
    wave   = np.zeros_like(t)

    for i, f in enumerate(freqs):
        phase = i * np.pi / 4
        # Her katman belirli hızda arpeggio yapar
        wave += (0.25 / (i + 1)) * np.sin(2 * np.pi * f * t + phase)
        # Düşük bas katmanı
        wave += 0.05 * np.sin(2 * np.pi * (f / 2) * t)

    # Fade in/out
    fade = int(sr * 1.0)
    wave[:fade]  *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    pcm = (wave * 32767 * 0.7).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, sr, pcm)
    return buf.getvalue()


def mix_audio(voice_bytes: bytes, music_bytes: bytes, duck: bool, duck_ratio: float) -> bytes:
    """
    Ses + müziği karıştırır. Ducking simülasyonu dahil.
    """
    def load_wav(b):
        buf = io.BytesIO(b)
        sr, data = wav.read(buf)
        return sr, data.astype(np.float32) / 32767

    vsr, v = load_wav(voice_bytes)
    msr, m = load_wav(music_bytes)

    # Aynı uzunluğa getir (müziği ses uzunluğuna kırp/döngüle)
    if len(m) < len(v):
        reps = (len(v) // len(m)) + 1
        m = np.tile(m, reps)
    m = m[:len(v)]

    # Ducking: ses aktifken müziği kıs
    if duck:
        win = int(vsr * 0.05)  # 50ms pencere
        for i in range(0, len(v) - win, win):
            rms = np.sqrt(np.mean(v[i:i+win]**2))
            if rms > 0.01:
                m[i:i+win] *= duck_ratio

    # Ses seviyeleri
    mixed = v * 1.0 + m * 0.35
    mixed = np.clip(mixed, -1.0, 1.0)

    pcm = (mixed * 32767 * 0.9).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, vsr, pcm)
    return buf.getvalue()


# ──────────────────────────────────────────────
# TÜRKÇE METİN ANALİZİ (lightweight)
# ──────────────────────────────────────────────

TONE_KEYWORDS = {
    "enerjik" : ["hızlı","dinamik","enerji","harika","heyecan","şimdi","bugün","fırsat","kampanya","indirim"],
    "kurumsal": ["güvenilir","profesyonel","kalite","lider","deneyim","çözüm","hizmet","teknoloji","sistem"],
    "dramatik": ["dönüşüm","devrim","tarihi","büyük","muhteşem","inanılmaz","özgür","kahraman","sınırsız"],
    "samimi"  : ["aile","sevgi","birlikte","paylaşım","mutluluk","sıcak","doğal","gerçek","güven"],
    "lüks"    : ["premium","lüks","özel","seçkin","zarif","sofistike","şık","prestij","marka"],
}

def analyze_text(text: str) -> dict:
    tl = text.lower()
    scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in TONE_KEYWORDS.items()}
    best   = max(scores, key=scores.get)
    total  = sum(scores.values()) or 1
    words  = len(text.split())
    return {
        "tone"      : best,
        "confidence": round(scores[best] / total, 2) if total > 0 else 0.0,
        "scores"    : scores,
        "words"     : words,
        "est_sec"   : round(words * 0.45, 1),
        "sentences" : text.count(".") + text.count("!") + text.count("?"),
    }


# ──────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎵 AI Müzik & Ses Prodüksiyon Stüdyosu</h1>
    <p>Türkçe Metin → Profesyonel Reklam Seslendirmesi + AI Arka Plan Müziği</p>
    <div>
        <span class="badge">🤖 XTTS v2 (Türkçe TTS)</span>
        <span class="badge">🎼 MusicGen (Meta AI)</span>
        <span class="badge">🎚️ Otomatik Ducking</span>
        <span class="badge">🎤 Ses Klonlama</span>
        <span class="badge">⚡ DEMO Modu</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Prodüksiyon Ayarları")

    tone = st.selectbox(
        "Reklam Tonu",
        ["enerjik", "kurumsal", "dramatik", "samimi", "lüks"],
        index=1,
    )

    TONE_DESC = {
        "enerjik" : "⚡ Hızlı, dinamik, coşkulu",
        "kurumsal": "🏢 Profesyonel, güven verici",
        "dramatik": "🎭 Duygusal, etkileyici, sinematik",
        "samimi"  : "💬 Sıcak, doğal, arkadaşça",
        "lüks"    : "💎 Zarif, sofistike, premium",
    }
    st.caption(TONE_DESC[tone])

    st.divider()
    duration   = st.slider("Müzik Süresi (sn)", 5, 60, 15, 5)
    speed      = st.slider("Konuşma Hızı", 0.7, 1.5, 1.0, 0.05)
    use_duck   = st.toggle("Ducking Aktif", True)
    duck_ratio = st.slider("Ducking Oranı", 0.1, 0.9, 0.30, 0.05, disabled=not use_duck)

    st.divider()
    st.markdown("### 📊 Sistem Durumu")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**TTS**")
        st.markdown(":orange[Demo]")
    with col_s2:
        st.markdown("**MusicGen**")
        st.markdown(":orange[Demo]")
    st.caption("Prodüksiyonda GPU ile gerçek model çalışır.")

    st.divider()
    st.markdown("### ℹ️ Demo Hakkında")
    st.info(
        "Bu demo **ML modeli yüklemeden** çalışır.\n\n"
        "Gerçek prodüksiyonda:\n"
        "- XTTS v2 ses sentezi\n"
        "- Meta MusicGen müzik\n"
        "- GPU ile ~5–15s üretim süresi"
    )

# ──────────────────────────────────────────────
# ANA SEKMELER
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Hızlı Pipeline",
    "🎙️ Ses Sentezi",
    "🎵 Müzik Üretimi",
    "📊 Metin Analizi",
])

# ═══════════════════════════════════════════════
# SEKME 1 – HIZLI PIPELINE
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("## ⚡ Generate & Mix — Tek Tıkla Tam Prodüksiyon")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        text_input = st.text_area(
            "Reklam Metni",
            value=(
                "Türkiye'nin lider yapay zeka şirketi TechNova, "
                "akıllı çözümleriyle iş dünyasını dönüştürüyor. "
                "Geleceği bugün keşfet."
            ),
            height=130,
            placeholder="Reklam metninizi buraya yazın…",
        )

        # Canlı analiz
        if text_input.strip():
            res = analyze_text(text_input)
            detected = res["tone"]
            conf     = res["confidence"]

            tone_colors = {
                "enerjik":"#FF6B35","kurumsal":"#6C63FF",
                "dramatik":"#C62828","samimi":"#2E7D32","lüks":"#6A1B9A"
            }
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Tespit Edilen Ton", detected.upper())
            with col_t2:
                st.metric("Güven", f"{conf*100:.0f}%")
            with col_t3:
                st.metric("Tahmini Süre", f"{res['est_sec']}s")

    with col_right:
        st.markdown("### Üretim Adımları")
        st.markdown("""
<div class="step-box"><span class="step-num">1</span> <b>Metin Analizi</b> — ton ve kategori tespiti</div>
<div class="step-box"><span class="step-num">2</span> <b>TTS Sentezi</b> — XTTS v2 Türkçe ses</div>
<div class="step-box"><span class="step-num">3</span> <b>AI Müzik</b> — MusicGen arka plan</div>
<div class="step-box"><span class="step-num">4</span> <b>Mix & Master</b> — Ducking + Kompresör</div>
""", unsafe_allow_html=True)

    st.divider()

    if st.button("⚡ GENERATE & MIX", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Lütfen reklam metni girin.")
        else:
            # Adım adım progress
            progress = st.progress(0)
            status   = st.empty()

            status.markdown("**[1/4]** Metin analiz ediliyor…")
            time.sleep(0.4)
            res = analyze_text(text_input)
            progress.progress(15)

            status.markdown("**[2/4]** 🎙️ Ses sentez ediliyor (XTTS v2 simülasyonu)…")
            time.sleep(0.8)
            voice_bytes = make_voice_wav(text_input, tone, speed)
            progress.progress(45)

            status.markdown(f"**[3/4]** 🎵 AI müzik üretiliyor ({duration}s, {tone} tonu)…")
            time.sleep(0.8)
            music_bytes = make_music_wav("", duration, tone)
            progress.progress(75)

            status.markdown("**[4/4]** 🎚️ Mix & Mastering uygulanıyor…")
            time.sleep(0.5)
            mixed_bytes = mix_audio(voice_bytes, music_bytes, use_duck, duck_ratio)
            progress.progress(100)
            status.markdown("✅ **Prodüksiyon tamamlandı!**")

            # Sonuç kartları
            st.divider()
            st.markdown("### 🎧 Sonuçlar")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("**🎙️ Seslendirme**")
                st.audio(voice_bytes, format="audio/wav")
                st.download_button("⬇️ Ses İndir", voice_bytes,
                                   f"seslendirme_{tone}.wav", "audio/wav",
                                   use_container_width=True)

            with r2:
                st.markdown("**🎵 Arka Plan Müziği**")
                st.audio(music_bytes, format="audio/wav")
                st.download_button("⬇️ Müzik İndir", music_bytes,
                                   f"muzik_{tone}_{duration}s.wav", "audio/wav",
                                   use_container_width=True)

            with r3:
                st.markdown("**🎚️ Final Mix**")
                st.audio(mixed_bytes, format="audio/wav")
                st.download_button("⬇️ Final İndir", mixed_bytes,
                                   f"final_mix_{tone}.wav", "audio/wav",
                                   use_container_width=True, type="primary")

            # Analiz özeti
            with st.expander("📊 Analiz Raporu"):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("**Ton Skorları**")
                    df = pd.DataFrame(list(res["scores"].items()),
                                      columns=["Ton", "Skor"]).sort_values("Skor", ascending=True)
                    st.bar_chart(df.set_index("Ton"))
                with col_a2:
                    st.markdown("**Üretim Özeti**")
                    st.markdown(f"""
| Parametre | Değer |
|-----------|-------|
| Tespit Edilen Ton | **{res['tone'].upper()}** |
| Güven | **{res['confidence']*100:.0f}%** |
| Kelime Sayısı | **{res['words']}** |
| Tahmini Konuşma Süresi | **{res['est_sec']}s** |
| Müzik Süresi | **{duration}s** |
| Ducking | **{'Aktif' if use_duck else 'Kapalı'}** |
| Ducking Oranı | **{duck_ratio:.0%}** |
                    """)


# ═══════════════════════════════════════════════
# SEKME 2 – SES SENTEZİ
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("## 🎙️ Ses Sentezi")
    st.caption("Metinden Türkçe seslendirme üret. Demo'da sine wave kullanılır; prodüksiyonda XTTS v2 çalışır.")

    c1, c2 = st.columns([2, 1])
    with c1:
        s_text = st.text_area("Seslendirme Metni", height=140,
                              value="Yeni nesil yapay zeka teknolojileriyle tanışın.",
                              key="s_text")
    with c2:
        s_tone  = st.selectbox("Ton Profili", ["enerjik","kurumsal","dramatik","samimi","lüks"],
                               index=1, key="s_tone")
        s_speed = st.slider("Hız", 0.7, 1.5, 1.0, 0.05, key="s_speed")
        s_lang  = st.selectbox("Dil", ["Türkçe (tr)","İngilizce (en)","Almanca (de)"], key="s_lang")

        st.markdown("**Ton Özellikleri:**")
        st.caption(TONE_DESC[s_tone])

    if st.button("🎙️ Ses Üret", type="primary", use_container_width=True, key="btn_synth"):
        with st.spinner("Ses sentez ediliyor…"):
            time.sleep(0.7)
            vb = make_voice_wav(s_text, s_tone, s_speed)
        st.success("✅ Ses üretildi!")
        st.audio(vb, format="audio/wav")
        st.download_button("⬇️ WAV İndir", vb, f"ses_{s_tone}.wav", "audio/wav",
                           use_container_width=True)

        # Dalga formu göster
        sr_demo, samples = wav.read(io.BytesIO(vb))
        wf = pd.DataFrame({"Genlik": samples[::50]})
        st.line_chart(wf, height=120)


# ═══════════════════════════════════════════════
# SEKME 3 – MÜZİK ÜRETİMİ
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("## 🎵 AI Arka Plan Müziği")
    st.caption("Metin tanımından müzik üret. Demo'da çok katmanlı sine wave; prodüksiyonda Meta MusicGen.")

    c1, c2 = st.columns([2, 1])
    with c1:
        m_prompt = st.text_area(
            "Müzik Tanımı (İngilizce en iyi sonucu verir)",
            value="modern minimalist electronic music, professional corporate, clean beats",
            height=100,
            key="m_prompt",
        )
        # Hazır şablonlar
        st.markdown("**Hazır Şablonlar:**")
        t_cols = st.columns(3)
        templates = {
            "💻 Teknoloji" : "modern minimalist electronic, futuristic synths, clean",
            "👗 Moda"      : "elegant fashion, trendy pop, light percussion",
            "🍽️ Gıda"     : "warm acoustic, cheerful, natural guitar",
            "🚗 Otomotiv"  : "powerful cinematic, strong bass, action",
            "💊 Sağlık"    : "calm wellness, soft piano, healing",
            "🏢 Kurumsal"  : "corporate background, professional strings",
        }
        for i, (label, tmpl) in enumerate(templates.items()):
            with t_cols[i % 3]:
                if st.button(label, key=f"t_{i}", use_container_width=True):
                    st.session_state["m_prompt_val"] = tmpl

    with c2:
        m_dur   = st.slider("Süre (sn)", 5, 60, 15, 5, key="m_dur")
        m_tone  = st.selectbox("Ek Ton", ["—","enerjik","kurumsal","dramatik","samimi","lüks"], key="m_tone")

        st.markdown("**MusicGen Model:**")
        st.caption("🟡 Demo: sine wave\n\n🟢 Prod: facebook/musicgen-medium")

    if st.button("🎵 Müzik Bestele", type="primary", use_container_width=True, key="btn_music"):
        with st.spinner(f"{m_dur}s müzik üretiliyor…"):
            time.sleep(0.8)
            tone_for_music = m_tone if m_tone != "—" else tone
            mb = make_music_wav(m_prompt, m_dur, tone_for_music)
        st.success("✅ Müzik üretildi!")
        st.audio(mb, format="audio/wav")
        st.download_button("⬇️ Müzik İndir", mb, f"muzik_{m_dur}s.wav", "audio/wav",
                           use_container_width=True)

        sr_m, samples_m = wav.read(io.BytesIO(mb))
        wf_m = pd.DataFrame({"Genlik": samples_m[::80]})
        st.line_chart(wf_m, height=120)


# ═══════════════════════════════════════════════
# SEKME 4 – METİN ANALİZİ
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Metin Analizi")
    st.caption("Metnin duygusal tonunu tespit et, müzik ve ses profili önerileri al.")

    a_text = st.text_area(
        "Analiz edilecek metin",
        value=(
            "Harika fırsatı kaçırma! Sınırlı süre için geçerli, "
            "süper indirimli kampanyamıza hemen katıl. "
            "Enerji dolu yeni sezon koleksiyonu şimdi çıktı!"
        ),
        height=150,
        key="a_text",
    )

    if st.button("🔍 Analiz Et", type="primary", use_container_width=True, key="btn_analyze"):
        with st.spinner("Analiz yapılıyor…"):
            time.sleep(0.3)
            r = analyze_text(a_text)

        st.divider()

        # Metrik kartlar
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Ton",         r["tone"].upper(), "#6C63FF"),
            ("Güven",       f"{r['confidence']*100:.0f}%", "#3ECFCF"),
            ("Kelime",      str(r["words"]), "#FF6B35"),
            ("Tahmini Süre",f"{r['est_sec']}s", "#4CAF50"),
        ]
        for col, (label, value, color) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f"""
<div class="metric-card">
    <div class="value" style="color:{color}">{value}</div>
    <div class="label">{label}</div>
</div>""", unsafe_allow_html=True)

        st.divider()
        c_chart, c_info = st.columns([1, 1])

        with c_chart:
            st.markdown("**Ton Dağılımı**")
            df_scores = pd.DataFrame(
                sorted(r["scores"].items(), key=lambda x: x[1], reverse=True),
                columns=["Ton", "Eşleşme Sayısı"]
            )
            st.bar_chart(df_scores.set_index("Ton"), height=220)

        with c_info:
            st.markdown("**Önerilen Ayarlar**")
            MUSIC_PROMPTS = {
                "enerjik" : "upbeat energetic electronic music, fast tempo, exciting",
                "kurumsal": "professional corporate background, calm piano strings",
                "dramatik": "dramatic orchestral, emotional, cinematic strings",
                "samimi"  : "warm friendly acoustic, gentle guitar, natural",
                "lüks"    : "elegant luxury jazz piano, sophisticated, smooth",
            }
            best = r["tone"]
            st.markdown(f"""
| Ayar | Öneri |
|------|-------|
| TTS Tonu | **{best.upper()}** |
| Konuşma Hızı | **{{"enerjik":1.15,"kurumsal":0.92,"dramatik":0.85,"samimi":1.0,"lüks":0.88}[best]}x** |
| Müzik Stili | _{MUSIC_PROMPTS[best]}_ |
| Ducking | **{'%30 (önerilen)' if best in ['kurumsal','dramatik','lüks'] else '%40 (dinamik)'}** |
            """)

            st.markdown("**Normalize Edilmiş Metin (TTS için):**")
            st.code(a_text.strip(), language="text")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; opacity:.5; font-size:.8rem; padding:.5rem">
AI Müzik & Ses Prodüksiyon Stüdyosu • Demo Modu •
Gerçek prodüksiyon için XTTS v2 + MusicGen + GPU gereklidir
</div>
""", unsafe_allow_html=True)
