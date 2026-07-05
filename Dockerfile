FROM python:3.12-slim

# Enforce Python environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create a secure non-root system group and user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Install python dependencies
COPY requirements/ /app/requirements/

RUN python -m pip install --upgrade pip && \
    python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements/prod.txt

# Copy project modules (excluding weight binaries via .dockerignore)
COPY src/ /app/src/
COPY config/ /app/config/

# Pre-create model mounting directory and adjust permissions
RUN mkdir -p /app/models && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
