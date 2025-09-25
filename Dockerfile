FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# OS packages needed by scikit-learn (libgomp1) and dvc (git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install your package so "from src ..." works
COPY setup.py .
COPY src ./src
RUN pip install --no-cache-dir -e .

# App code
COPY app ./app

# DVC metadata so `dvc pull` can find the remote (optional but recommended)
COPY .dvc ./.dvc
COPY dvc.yaml dvc.lock ./

# Copy artifacts directly into the Docker image
COPY artifacts/ ./artifacts/

EXPOSE 8501

# Run the app directly; the app fetches artifacts via boto3 if missing
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]