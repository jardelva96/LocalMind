# ===== Runtime =====
FROM python:3.11-slim

# Dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Diretório de app
WORKDIR /app

# Copia só o necessário (mais rápido no build)
COPY pyproject.toml /app/
COPY src /app/src
COPY models/export /app/models/export
COPY configs /app/configs

# Instalar dependências mínimas para servir ONNX
# (fastapi, uvicorn, onnxruntime; httpx para o healthcheck opcional)
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.22" "onnxruntime>=1.16" httpx

# Instalar o pacote local (modo editable opcional)
RUN pip install -e .

# Porta e comando
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# Variáveis de caminho default (pode sobrescrever no docker run)
ENV IAZERO_MODEL=/app/models/export/iazero.onnx
ENV IAZERO_PREPROC=/app/models/export/preprocess.json
ENV IAZERO_HOST=0.0.0.0
ENV IAZERO_PORT=8000

CMD ["python","-m","iac_core.serve_onnx","--model","/app/models/export/iazero.onnx","--preprocess","/app/models/export/preprocess.json","--host","0.0.0.0","--port","8000"]
