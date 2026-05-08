# ── Base: CUDA 12.6 + cuDNN на Ubuntu 22.04 (совместимо с RTX 4050 / любой NVIDIA) ──
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Системные зависимости + Python 3.11 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        curl \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── uv — быстрый пакетный менеджер ──
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ── Копируем только файлы, нужные для установки зависимостей (кеш слоя) ──
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# ── Создаём venv и устанавливаем все зависимости (PyTorch cu126 из pyproject.toml) ──
RUN uv venv --python 3.11 .venv \
    && uv pip install --python .venv/bin/python -e ".[server]"

ENV PATH="/app/.venv/bin:$PATH"

# ── Копируем остальной код (server.py, etc.) ──
COPY server.py ./

# ── Модели монтируются снаружи через volume, не копируем в образ ──
VOLUME ["/app/models"]

EXPOSE 4396

# Запуск: стандартный GPU-режим.
# Для INT8: docker run ... rotate-captcha-crack python server.py --quant
CMD ["python", "server.py"]
