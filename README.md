# 🎵 AI Müzik & Ses Prodüksiyon Stüdyosu

> **Türkçe öncelikli** TTS · Ses Klonlama · AI Müzik Üretimi · Otomatik Prodüksiyon

---

## 📐 Mimari Akış

```
Kullanıcı Girdisi (Metin + Ton Seçimi)
          │
          ▼
┌─────────────────────┐
│   TextAnalyzer      │  → Duygu analizi, kategori tespiti
│   TurkishUtils      │  → UTF-8 normalleştirme, kısaltma açma
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌──────────────┐
│VoiceEngine│  │MusicGenerator│
│ XTTS v2  │  │  MusicGen    │
│ (Türkçe) │  │  (Meta AI)   │
└────┬─────┘  └──────┬───────┘
     │               │
     │  voice.wav    │  music.wav
     └───────┬───────┘
             ▼
     ┌───────────────┐
     │  AudioMixer   │
     │  + Ducking    │  ← Ses konuşurken müzik %30'a düşer
     │  + Mastering  │
     └───────┬───────┘
             ▼
       final_mix.wav
     (İndir / Oynat)
```

---

## 🗂️ Proje Yapısı

```
AI-Music-Production/
├── main.py                     # Ana giriş noktası (API / UI / Pipeline)
├── app.py                      # Streamlit web arayüzü
├── config.py                   # Merkezi konfigürasyon
├── requirements.txt            # Python bağımlılıkları
├── Dockerfile                  # GPU destekli container
├── docker-compose.yml          # API + UI servisleri
├── .env.example                # Ortam değişkenleri şablonu
│
├── models/
│   ├── voice_engine.py         # XTTS v2 Türkçe TTS motoru
│   ├── music_generator.py      # MusicGen AI müzik üretimi
│   └── voice_cloner.py         # Ses klonlama & profil yönetimi
│
├── utils/
│   ├── audio_processor.py      # Format dönüştürme, normalize, trim
│   ├── mixer.py                # Mix, ducking, mastering
│   ├── text_analyzer.py        # Duygu analizi, ton tespiti
│   ├── file_manager.py         # Dosya yükleme, temizlik
│   └── turkish_utils.py        # Türkçe karakter, sayı → yazı
│
├── api/
│   └── routes.py               # FastAPI REST endpoint'leri
│
└── uploads/
    ├── voices/                 # Kullanıcı ses profilleri
    └── outputs/                # Üretilen dosyalar
```

---

## 🚀 Kurulum

### 1. Ortam Değişkenleri

```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

### 2. Python Ortamı

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# GPU (CUDA 11.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# CPU (daha yavaş):
pip install -r requirements.txt
```

### 3. Docker ile Başlatma (Önerilen)

```bash
# Tüm servisleri başlat (API + UI)
docker-compose up --build

# Yalnızca API:
docker-compose up api

# Yalnızca UI:
docker-compose up ui
```

### 4. Manuel Başlatma

```bash
# API sunucusu (port 8000)
python main.py api

# Streamlit arayüzü (port 8501)
python main.py ui

# Komut satırı pipeline
python main.py pipeline \
    --text "Türkiye'nin lider yapay zeka şirketi." \
    --tone enerjik \
    --duration 30 \
    --output reklam.wav
```

---

## 🔌 API Referansı

### Base URL: `http://localhost:8000`

| Metod | Endpoint | Açıklama |
|-------|----------|----------|
| `GET` | `/health` | Sistem durumu |
| `POST` | `/synthesize` | Metni sese çevir |
| `POST` | `/generate-music` | Müzik üret |
| `POST` | `/mix` | Ses + müziği karıştır |
| `POST` | `/full-pipeline` | **Tek istekle tam prodüksiyon** |
| `POST` | `/voices/register` | Ses profili kaydet |
| `GET` | `/voices` | Profil listesi |
| `DELETE` | `/voices/{id}` | Profil sil |
| `GET` | `/outputs` | Çıktı dosyaları |
| `GET` | `/download/{file}` | Dosya indir |
| `POST` | `/analyze-text` | Metin analizi |

### Örnek: Tam Pipeline

```bash
curl -X POST http://localhost:8000/full-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Türkiye'\''nin lider teknoloji şirketi, yapay zeka ile geleceği bugün inşa ediyor.",
    "tone_profile": "kurumsal",
    "duration": 30
  }'
```

```json
{
  "status": "ok",
  "file": "final_mix_20240115_143022_a1b2c3d4.wav",
  "download": "/download/final_mix_20240115_143022_a1b2c3d4.wav",
  "analysis": {
    "tone": "kurumsal",
    "confidence": 0.78,
    "category": "teknoloji",
    "stats": { "word_count": 13, "estimated_sec": 5.8 }
  }
}
```

### Örnek: Ses Profili Kaydet

```bash
curl -X POST http://localhost:8000/voices/register \
  -F "file=@/path/to/ornek_ses.wav" \
  -F "name=Mehmet Yılmaz" \
  -F "description=Erkek, kurumsal, 35 yaş"
```

---

## ⚙️ Konfigürasyon

### Reklam Ton Profilleri

| Ton | Hız | Müzik Stili |
|-----|-----|-------------|
| `enerjik` | 1.15x | Upbeat electronic, fast tempo |
| `kurumsal` | 0.92x | Professional piano strings |
| `dramatik` | 0.85x | Cinematic orchestral |
| `samimi` | 1.00x | Warm acoustic, guitar |
| `lüks` | 0.88x | Elegant jazz piano |

### MusicGen Model Seçimi

```env
# Hız önceliği (~300MB)
MUSIC_MODEL=facebook/musicgen-small

# Denge (~1.5GB)
MUSIC_MODEL=facebook/musicgen-medium

# Kalite önceliği (~3.3GB)
MUSIC_MODEL=facebook/musicgen-large
```

---

## ☁️ Bulut Kurulum

### Google Cloud Platform (GCP)

```bash
# 1. NVIDIA T4 GPU'lu VM oluştur
gcloud compute instances create ai-music-vm \
  --zone=europe-west4-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE

# 2. NVIDIA sürücülerini kur
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py | sudo python3

# 3. Docker + NVIDIA Container Toolkit kur
sudo apt install docker.io -y
sudo systemctl enable --now docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit -y
sudo systemctl restart docker

# 4. Uygulamayı başlat
git clone https://github.com/yasingokcek/aimusic.git
cd aimusic
cp .env.example .env
docker-compose up --build -d
```

### AWS SageMaker

```python
# SageMaker endpoint oluşturma (örnek)
import sagemaker
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data   = "s3://bucket/model.tar.gz",
    role         = "SageMakerRole",
    framework_version = "2.1",
    py_version   = "py310",
    entry_point  = "main.py",
)
predictor = model.deploy(
    instance_type  = "ml.g4dn.xlarge",  # T4 GPU
    initial_instance_count = 1,
)
```

---

## 🔒 Güvenlik ve Etik

- **Telif Hakkı**: Ses klonlama yalnızca kendi sesiniz veya açık rıza alınan sesler için kullanılmalıdır. `VoiceCloner._check_copyright()` hash tabanlı temel kontrol sağlar; production'da [ACRCloud](https://www.acrcloud.com/) entegrasyonu önerilir.
- **Veri Gizliliği**: Ses dosyaları sunucuda şifreli disk üzerinde saklanmalıdır.
- **API Güvenliği**: `.env` dosyasında `SECRET_KEY` değiştirilmeli ve Bearer token kimlik doğrulaması eklenmelidir.

---

## 📦 GPU Gereksinimleri

| Model Kombinasyonu | VRAM |
|--------------------|------|
| XTTS v2 + MusicGen Small | ~6 GB |
| XTTS v2 + MusicGen Medium | ~10 GB |
| XTTS v2 + MusicGen Large | ~14 GB |

Önerilen: **NVIDIA T4 (16GB)** veya **A100 (40GB)**
