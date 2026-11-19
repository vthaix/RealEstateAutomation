# Ảnh ứng dụng: Rule Engine + Knowledge Graph + LLM parser + API/Web (không gồm PhoBERT/torch cho nhẹ)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY docker ./docker
COPY tests ./tests
COPY pyproject.toml .env.example ./

EXPOSE 8000
CMD ["python", "docker/entrypoint.py"]
