# -*- coding: utf-8 -*-
"""
start_tunnel.py – Streamlit + Ngrok Tünel Başlatıcı
=====================================================
Bu script'i çalıştırınca:
1. Streamlit demo'yu arka planda başlatır
2. ngrok tüneli açar
3. Herkese açık URL'yi ekrana yazar

Çalıştır:
    python start_tunnel.py

ngrok token gerekliyse:
    ngrok authtoken <YOUR_TOKEN>
    (https://dashboard.ngrok.com/get-started/your-authtoken)
"""

import os
import sys
import time
import subprocess

def main():
    # 1. Streamlit başlat
    print("🚀 Streamlit başlatılıyor (port 8501)…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "demo.py",
         "--server.port", "8501",
         "--server.headless", "true",
         "--server.address", "0.0.0.0",
         "--server.fileWatcherType", "none"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    print(f"  ✅ Streamlit PID: {proc.pid}")

    # 2. ngrok tüneli aç
    try:
        from pyngrok import ngrok, conf  # type: ignore

        # Token varsa ayarla (isteğe bağlı)
        ngrok_token = os.getenv("NGROK_AUTHTOKEN", "")
        if ngrok_token:
            conf.get_default().auth_token = ngrok_token

        print("🌐 ngrok tüneli açılıyor…")
        tunnel = ngrok.connect(8501, "http")
        public_url = tunnel.public_url

        print("\n" + "="*55)
        print("  🎵 AI Müzik Prodüksiyon Demo — CANLI!")
        print("="*55)
        print(f"\n  👉  {public_url}\n")
        print("  Bu URL'yi tarayıcıda aç. Herkese açık!")
        print("  Çıkmak için Ctrl+C\n")
        print("="*55)

        # Streamlit kapanana kadar bekle
        proc.wait()

    except ImportError:
        print("\n⚠️  pyngrok kurulu değil. Alternatif yöntemler:")
        print("  pip install pyngrok")
        print("\nVeya şu URL'leri dene:")
        print("  http://localhost:8501         (aynı ağdaysan)")
        print("  http://0.0.0.0:8501")

    except KeyboardInterrupt:
        print("\n⛔ Durduruldu.")
        proc.terminate()

if __name__ == "__main__":
    main()
