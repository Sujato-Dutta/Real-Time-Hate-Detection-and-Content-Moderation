# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and app
COPY src ./src
COPY app ./app

# Artifacts (optional): keep if you want the model inside the image; otherwise mount at runtime
COPY artifacts ./artifacts

# Config (if needed at runtime)
COPY src/config/config.yaml ./src/config/config.yaml

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]