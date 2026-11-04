FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU only torch keeps the image small. It satisfies the torch pins in requirements.txt
COPY requirements.txt .
RUN grep -E '^(torch|torchvision)==' requirements.txt > torch-requirements.txt \
    && pip install --no-cache-dir -r torch-requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

# download the weights now so containers do not download on start.
# the voice model comes from models/config.json so it always matches the trained models
ENV HF_HOME=/app/hf-cache
RUN python -c "import json, sys; sys.path.insert(0, 'src'); \
import features; \
config = json.load(open('models/config.json')); \
features._load_clip(); \
features._load_voice_pipeline(config['voice_model_id'])"
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
# no access log, it would store client IPs. api.py logs requests without them
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
