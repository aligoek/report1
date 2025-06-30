# Bir Python 3.9 veya daha yüksek sürümü tabanlı imaj kullanın
FROM python:3.9-slim-buster

# Çalışma dizinini ayarlayın
WORKDIR /app

# Gerekli bağımlılıkları kopyalayın ve yükleyin
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Font dizinini ve font dosyasını kopyalayın
COPY fonts /app/fonts

# Uygulama kodunu kopyalayın
COPY report_with_api.py .
COPY .env .

# API anahtarının bir ortam değişkeni olarak ayarlandığından emin olun
# DİKKAT: Üretim ortamında bu şekilde hassas bilgileri Dockerfile'a gömmek yerine
# Docker Secrets veya Kubernetes Secrets kullanılması önerilir.
# Ancak çevrimdışı test için bu kabul edilebilir.
# ENV GEMINI_API_KEY="YOUR_GEMINI_API_KEY" # Burayı .env dosyasından okuyacağız

# Uygulamayı çalıştırın
CMD ["uvicorn", "report_with_api:app", "--host", "0.0.0.0", "--port", "8000"]