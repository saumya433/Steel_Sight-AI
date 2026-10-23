# CPU image for the dashboard / edge box without a GPU.
# For an NVIDIA edge device start from nvcr.io/nvidia/pytorch instead and export a TensorRT engine.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .

ENV STEELSIGHT_WEIGHTS=models/best_v1.pt \
    STEELSIGHT_BANK=models/anomaly_bank.pt \
    STEELSIGHT_DB=data/steelsight.db
EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
