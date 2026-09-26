FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
