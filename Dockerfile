FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY models/ ./models/

# Pre-load models at build time for faster startup
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models/ner

EXPOSE 8000

# Uvicorn with multiple workers for concurrent request handling
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
