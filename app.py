# -*- coding: utf-8 -*-
"""
app.py – Streamlit Web Arayüzü
================================
AI Müzik & Ses Prodüksiyon Stüdyosu kullanıcı arayüzü.

Bölümler:
  🎙️ Ses Sentezi  – Metin gir, ton seç, ses üret
  🎵 Müzik Üretim – Prompt gir, stil seç, müzik üret
  🎚️ Prodüksiyon  – Ses + müziği otomatik karıştır
  📂 Ses Kütüphanesi – Profil yükle, yönet, klonla
  📊 Analiz       – Metin analizi ve öneri paneli

Çalıştırma:
  streamlit run app.py
  (veya: python main.py ui)
"""

# ──────────────────────────────────────────────
# ENCODING
# ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import logging
from pathlib import Path

# ──────────────────────────────────────────────
# STREAMLIT SAYFA YAPISI
# ──────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Müzik & Ses Prodüksiyon Stüdyosu",
    page_icon  = "🎵",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────
# ÖZEL CSS (Türkçe karakter desteği + tema)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Ana başlık */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Kart kutuları */
    .card {
        background: #1E1E2E;
        border: 1px solid #313153;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Durum rozetleri */
    .badge-success { color: #4CAF50; font-weight: 600; }
    .badge-warning { color: #FF9800; font-weight: 600; }
    .badge-error   { color: #F44336; font-weight: 600; }

    /* Türkçe karakter – dosya yükleme alanı */
    .stFileUploader label { font-size: 0.9rem; }

    /* Progress bar rengi */
    .stProgress > div > div { background: #6C63FF; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# YARDIMCI FONKSIYONLAR
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Model yükleniyor…")
def get_voice_engine():
    from models import VoiceEngine
    return VoiceEngine()

@st.cache_resource(show_spinner="Müzik modeli yükleniyor…")
def get_music_generator():
    from models import MusicGenerator
    return MusicGenerator()

@st.cache_resource
def get_voice_cloner():
    from models import VoiceCloner
    return VoiceCloner()

@st.cache_resource
def get_mixer():
    from utils import AudioMixer
    return AudioMixer()

@st.cache_resource
def get_text_analyzer():
    from utils import TextAnalyzer
    return TextAnalyzer()

@st.cache_resource
def get_file_manager():
    from utils import FileManager
    return FileManager()


def render_audio_player(file_path: str, label: str = "Dinle"):
    """Ses oynatıcı render et."""
    if Path(file_path).exists():
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            label        = f"⬇️ {label} (.wav)",
            data         = audio_bytes,
            file_name    = Path(file_path).name,
            mime         = "audio/wav",
            use_container_width = True,
        )


def show_analysis_card(analysis: dict):
    """Metin analiz sonuçlarını göster."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tespit Edilen Ton", analysis["tone"].upper())
        st.metric("Güven Skoru", f"{analysis['confidence']*100:.0f}%")
    with col2:
        st.metric("Kelime Sayısı", analysis["stats"]["word_count"])
        st.metric("Tahmini Süre", f"{analysis['stats']['estimated_sec']}s")
    with col3:
        st.metric("Kategori", analysis["category"].upper())
        st.metric("Cümle Sayısı", analysis["stats"]["sentence_count"])


# ──────────────────────────────────────────────
# ANA BAŞLIK
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎵 AI Müzik & Ses Prodüksiyon Stüdyosu</h1>
    <p>Türkçe Metin → Profesyonel Reklam Seslendirmesi + AI Müzik</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR – HIZLI AYARLAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/GPU-Aktif-brightgreen", use_container_width=False)
    st.markdown("---")
    st.markdown("### ⚙️ Genel Ayarlar")

    global_tone = st.selectbox(
        "Varsayılan Reklam Tonu",
        options    = ["enerjik", "kurumsal", "dramatik", "samimi", "lüks"],
        index      = 1,
        key        = "global_tone",
        help       = "Tüm üretimler için varsayılan ton",
    )

    global_duration = st.slider(
        "Müzik Süresi (sn)",
        min_value = 10, max_value = 120, value = 30, step = 5,
        key       = "global_duration",
    )

    use_ducking = st.toggle("Ducking Aktif", value=True,
                             help="Ses konuşurken müziği otomatik kıs (%30)")
    duck_ratio  = st.slider("Ducking Oranı", 0.1, 0.9, 0.3, 0.05,
                             disabled=not use_ducking)

    st.markdown("---")
    st.markdown("### 📊 Sistem Durumu")
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        st.markdown(f"GPU: {'✅ Aktif' if gpu_ok else '⚠️ CPU modu'}")
        if gpu_ok:
            st.caption(torch.cuda.get_device_name(0))
    except ImportError:
        st.markdown("GPU: ⚠️ PyTorch kurulu değil")


# ──────────────────────────────────────────────
# SEKMELER
# ──────────────────────────────────────────────
tab_synth, tab_music, tab_prod, tab_voices, tab_analyze = st.tabs([
    "🎙️ Ses Sentezi",
    "🎵 Müzik Üretimi",
    "🎚️ Prodüksiyon",
    "📂 Ses Kütüphanesi",
    "📊 Metin Analizi",
])

# ═══════════════════════════════════════════════
# SEKME 1 – SES SENTEZİ
# ═══════════════════════════════════════════════
with tab_synth:
    st.markdown("## 🎙️ Türkçe Ses Sentezi")
    st.caption("Metninizi girin, ton ve ses profili seçin.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        synth_text = st.text_area(
            "Seslendirme Metni",
            placeholder = "Buraya reklam metninizi Türkçe olarak girin…\n\nÖrnek: Türkiye'nin lider yazılım şirketi TechCorp, yeni yapay zeka çözümleriyle iş dünyasını dönüştürüyor.",
            height      = 160,
            key         = "synth_text",
        )

    with col_right:
        synth_tone = st.selectbox(
            "Reklam Tonu",
            options = ["enerjik", "kurumsal", "dramatik", "samimi", "lüks"],
            index   = ["enerjik", "kurumsal", "dramatik", "samimi", "lüks"].index(global_tone),
            key     = "synth_tone",
        )

        synth_speed = st.slider("Konuşma Hızı", 0.5, 2.0, 1.0, 0.05, key="synth_speed")
        synth_lang  = st.selectbox("Dil", ["tr", "en", "de", "fr", "es"], key="synth_lang")

        # Kayıtlı ses seçimi
        try:
            cloner  = get_voice_cloner()
            voices  = cloner.list_voices()
            v_opts  = ["Varsayılan Ses"] + [f"{v['name']} ({v['id']})" for v in voices]
            v_sel   = st.selectbox("Ses Profili", v_opts, key="synth_voice")
            voice_id = None if v_sel == "Varsayılan Ses" else voices[v_opts.index(v_sel) - 1]["id"]
        except Exception:
            voice_id = None
            st.caption("Ses profili yüklenemedi.")

    if st.button("🎙️ Sesi Üret", type="primary", use_container_width=True, key="btn_synth"):
        if not synth_text.strip():
            st.error("Lütfen bir metin girin.")
        else:
            with st.spinner("Ses üretiliyor…"):
                try:
                    from utils.turkish_utils import TurkishUtils
                    fm   = get_file_manager()
                    ve   = get_voice_engine()
                    cl   = get_voice_cloner()
                    norm = TurkishUtils.normalize_for_tts(synth_text)

                    speaker_wav = None
                    if voice_id:
                        speaker_wav = cl.get_voice_path(voice_id)

                    out_path = fm.new_output_path("voice")
                    ve.synthesize(
                        text         = norm,
                        output_path  = out_path,
                        speaker_wav  = speaker_wav,
                        language     = synth_lang,
                        tone_profile = synth_tone,
                        speed        = synth_speed,
                    )
                    st.success("✅ Ses üretildi!")
                    st.session_state["last_voice"] = out_path
                    render_audio_player(out_path, "Ses Dosyasını İndir")
                except Exception as exc:
                    st.error(f"Hata: {exc}")

# ═══════════════════════════════════════════════
# SEKME 2 – MÜZİK ÜRETİMİ
# ═══════════════════════════════════════════════
with tab_music:
    st.markdown("## 🎵 AI Arka Plan Müziği")
    st.caption("Metin tanımıyla özgün müzik bestele.")

    col_ml, col_mr = st.columns([2, 1])

    with col_ml:
        music_prompt = st.text_area(
            "Müzik Tanımı",
            value       = "modern minimalist elektronik müzik, teknoloji reklamı için, temiz ve profesyonel",
            height      = 100,
            key         = "music_prompt",
            help        = "İngilizce tanım en iyi sonucu verir. Türkçe de desteklenir.",
        )

        # Hazır şablonlar
        st.markdown("**Hazır Şablonlar:**")
        template_cols = st.columns(3)
        templates = {
            "Teknoloji"  : "modern minimalist electronic music, futuristic synths, clean beats, professional",
            "Moda"       : "elegant fashion background, trendy pop, light percussion, sophisticated",
            "Gıda"       : "warm acoustic background, appetizing cheerful, light guitar, natural",
            "Otomotiv"   : "powerful cinematic, strong bass, driving rhythm, action",
            "Sağlık"     : "calm wellness music, soft piano, healing, nature sounds",
            "Kurumsal"   : "corporate background, professional piano strings, neutral, clean",
        }
        for i, (name, tmpl) in enumerate(templates.items()):
            with template_cols[i % 3]:
                if st.button(name, key=f"tmpl_{name}", use_container_width=True):
                    st.session_state["music_prompt_val"] = tmpl
                    st.rerun()

    with col_mr:
        music_dur  = st.slider("Süre (sn)", 10, 120, global_duration, 5, key="music_dur")
        music_tone = st.selectbox(
            "Reklam Tonu (ek renk)",
            ["—", "enerjik", "kurumsal", "dramatik", "samimi", "lüks"],
            key = "music_tone",
        )

    if st.button("🎵 Müzik Bestele", type="primary", use_container_width=True, key="btn_music"):
        if not music_prompt.strip():
            st.error("Lütfen müzik tanımı girin.")
        else:
            with st.spinner(f"Müzik üretiliyor ({music_dur}s)… Bu işlem GPU'da ~10–30s sürebilir."):
                try:
                    fm   = get_file_manager()
                    mg   = get_music_generator()
                    tone = None if music_tone == "—" else music_tone
                    out  = fm.new_output_path("music")
                    mg.generate(
                        prompt       = music_prompt,
                        output_path  = out,
                        duration     = music_dur,
                        tone_profile = tone,
                    )
                    st.success("✅ Müzik üretildi!")
                    st.session_state["last_music"] = out
                    render_audio_player(out, "Müzik Dosyasını İndir")
                except Exception as exc:
                    st.error(f"Hata: {exc}")

# ═══════════════════════════════════════════════
# SEKME 3 – PRODÜKSİYON (MİX)
# ═══════════════════════════════════════════════
with tab_prod:
    st.markdown("## 🎚️ Otomatik Prodüksiyon & Mix")
    st.caption("Ses ve müziği birleştir. Ducking ile profesyonel karışım.")

    st.markdown("### Hızlı Pipeline")
    st.info(
        "Metin gir → Tonu seç → **Generate & Mix** butonuna bas → "
        "Nihai .wav dosyasını dinle/indir."
    )

    pipeline_text = st.text_area(
        "Reklam Metni",
        placeholder = "Seslendirme metnini buraya girin…",
        height      = 140,
        key         = "pipeline_text",
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        pipe_tone  = st.selectbox("Reklam Tonu", ["auto", "enerjik", "kurumsal", "dramatik", "samimi", "lüks"],
                                   key="pipe_tone", help="auto = metinden otomatik tespit")
    with col_p2:
        pipe_dur   = st.slider("Müzik Süresi (sn)", 10, 120, global_duration, 5, key="pipe_dur")
    with col_p3:
        pipe_voice = st.selectbox(
            "Ses Profili",
            ["Varsayılan"] + [f"{v['name']}" for v in (get_voice_cloner().list_voices() if True else [])],
            key = "pipe_voice",
        )

    st.markdown("---")
    st.markdown("### Veya: Manuel Mix (mevcut dosyalar)")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.caption("Son üretilen ses:")
        if "last_voice" in st.session_state:
            render_audio_player(st.session_state["last_voice"], "Ses")
        else:
            st.info("Henüz ses üretilmedi.")

    with col_m2:
        st.caption("Son üretilen müzik:")
        if "last_music" in st.session_state:
            render_audio_player(st.session_state["last_music"], "Müzik")
        else:
            st.info("Henüz müzik üretilmedi.")

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("⚡ Generate & Mix (Tam Pipeline)", type="primary",
                     use_container_width=True, key="btn_pipeline"):
            if not pipeline_text.strip():
                st.error("Lütfen reklam metni girin.")
            else:
                prog = st.progress(0, "Hazırlanıyor…")
                try:
                    from utils.turkish_utils import TurkishUtils
                    fm   = get_file_manager()
                    ve   = get_voice_engine()
                    mg   = get_music_generator()
                    mx   = get_mixer()
                    ta   = get_text_analyzer()
                    cl   = get_voice_cloner()

                    norm     = TurkishUtils.normalize_for_tts(pipeline_text)
                    analysis = ta.full_analysis(norm)
                    tone     = pipe_tone if pipe_tone != "auto" else analysis["tone"]
                    prompt   = analysis["music_prompt"]

                    prog.progress(10, "Metin analiz edildi…")

                    # TTS
                    vpath = fm.new_output_path("voice_raw")
                    ve.synthesize(text=norm, output_path=vpath, tone_profile=tone)
                    prog.progress(40, "Ses sentezi tamamlandı…")

                    # Müzik
                    mpath = fm.new_output_path("music_raw")
                    mg.generate(prompt=prompt, output_path=mpath, duration=pipe_dur, tone_profile=tone)
                    prog.progress(75, "Müzik üretildi…")

                    # Mix
                    fpath = fm.new_output_path("final_mix")
                    mx.mix(voice_path=vpath, music_path=mpath, output_path=fpath,
                           ducking=use_ducking, duck_ratio=duck_ratio)
                    prog.progress(100, "✅ Tamamlandı!")

                    st.session_state["last_voice"] = vpath
                    st.session_state["last_music"] = mpath
                    st.session_state["last_final"] = fpath

                    st.success("✅ Prodüksiyon tamamlandı!")
                    show_analysis_card(analysis)
                    st.markdown("### 🎧 Nihai Üretim")
                    render_audio_player(fpath, "Nihai Prodüksiyonu İndir")

                except Exception as exc:
                    st.error(f"Pipeline hatası: {exc}")

    with btn_col2:
        if st.button("🎚️ Manuel Mix", use_container_width=True, key="btn_manual_mix"):
            v = st.session_state.get("last_voice")
            m = st.session_state.get("last_music")
            if not v or not m:
                st.error("Önce ses ve müzik üretin.")
            else:
                with st.spinner("Mix yapılıyor…"):
                    try:
                        fm    = get_file_manager()
                        mx    = get_mixer()
                        fpath = fm.new_output_path("manual_mix")
                        mx.mix(voice_path=v, music_path=m, output_path=fpath,
                               ducking=use_ducking, duck_ratio=duck_ratio)
                        st.success("✅ Mix tamamlandı!")
                        render_audio_player(fpath, "Mix Dosyasını İndir")
                    except Exception as exc:
                        st.error(f"Mix hatası: {exc}")

# ═══════════════════════════════════════════════
# SEKME 4 – SES KÜTÜPHANESİ
# ═══════════════════════════════════════════════
with tab_voices:
    st.markdown("## 📂 Ses Kütüphanesi ve Klonlama")

    col_vl, col_vr = st.columns([1, 1])

    with col_vl:
        st.markdown("### ➕ Yeni Ses Profili Ekle")
        st.info(
            "En iyi sonuç için:\n"
            "- En az 30 saniye, tercihen 1–5 dakika\n"
            "- Arka plan gürültüsü olmayan ortam\n"
            "- Net ve doğal diksiyonla kayıt"
        )

        v_name = st.text_input("Profil Adı", placeholder="Örnek: Ahmet Yılmaz – Kurumsal", key="v_name")
        v_desc = st.text_area("Açıklama (isteğe bağlı)", height=60, key="v_desc")
        v_file = st.file_uploader(
            "Ses Dosyası (.wav, .mp3, .ogg, .flac)",
            type    = ["wav", "mp3", "ogg", "flac", "m4a"],
            key     = "v_file",
        )

        if st.button("💾 Profili Kaydet", type="primary", use_container_width=True, key="btn_save_voice"):
            if not v_name.strip():
                st.error("Profil adı boş olamaz.")
            elif not v_file:
                st.error("Ses dosyası seçin.")
            else:
                with st.spinner("Profil kaydediliyor ve doğrulanıyor…"):
                    try:
                        fm      = get_file_manager()
                        cl      = get_voice_cloner()
                        content = v_file.read()
                        tmp     = fm.save_upload(content, v_file.name)
                        profile = cl.register_voice(
                            audio_path  = tmp,
                            voice_name  = v_name,
                            description = v_desc,
                        )
                        Path(tmp).unlink(missing_ok=True)
                        st.success(f"✅ Profil kaydedildi! ID: `{profile['id']}`")
                        st.json(profile)
                    except Exception as exc:
                        st.error(f"Hata: {exc}")

    with col_vr:
        st.markdown("### 🗂️ Kayıtlı Profiller")
        try:
            cl     = get_voice_cloner()
            voices = cl.list_voices()
            if not voices:
                st.info("Henüz kayıtlı ses profili yok.")
            else:
                for v in voices:
                    with st.expander(f"🎤 {v['name']} — ID: `{v['id']}`"):
                        st.write(f"**Süre:** {v['duration']}s")
                        st.write(f"**Örnekleme:** {v['sample_rate']} Hz")
                        st.write(f"**Açıklama:** {v.get('description', '—')}")
                        if st.button(f"🗑️ Sil", key=f"del_{v['id']}"):
                            cl.delete_voice(v["id"])
                            st.success("Profil silindi.")
                            st.rerun()
        except Exception as exc:
            st.error(f"Profiller yüklenemedi: {exc}")

# ═══════════════════════════════════════════════
# SEKME 5 – METİN ANALİZİ
# ═══════════════════════════════════════════════
with tab_analyze:
    st.markdown("## 📊 Metin Analizi ve Öneri Paneli")
    st.caption("Metninizi analiz edin, sistem otomatik ton ve müzik stili önerir.")

    analysis_text = st.text_area(
        "Analiz Edilecek Metin",
        placeholder = "Reklam metninizi buraya yapıştırın…",
        height      = 180,
        key         = "analysis_text",
    )

    if st.button("🔍 Analiz Et", type="primary", use_container_width=True, key="btn_analyze"):
        if not analysis_text.strip():
            st.error("Lütfen metin girin.")
        else:
            from utils.turkish_utils import TurkishUtils
            ta   = get_text_analyzer()
            norm = TurkishUtils.normalize_for_tts(analysis_text)
            res  = ta.full_analysis(norm)

            st.markdown("### Analiz Sonuçları")
            show_analysis_card(res)

            st.markdown("### Ton Dağılımı")
            import pandas as pd
            scores_df = pd.DataFrame(
                list(res["scores"].items()), columns=["Ton", "Skor"]
            ).sort_values("Skor", ascending=False)
            st.bar_chart(scores_df.set_index("Ton"))

            st.markdown("### Önerilen Müzik Prompt'u")
            st.code(res["music_prompt"], language="text")

            st.markdown("### Normalize Edilmiş Metin")
            st.text_area("TTS'e gönderilecek metin:", norm, height=100, key="norm_out")
