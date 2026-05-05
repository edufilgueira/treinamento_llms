# Runpod Serverless: handler em streaming (yield SSE do llama-server) + return_aggregate_stream;
# Oráculo consome GET .../stream/{job_id}. Base: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# Esta imagem NÃO inclui o .gguf (pull rápido). No Runpod: Network Volume montado em /models (ou outro path)
# e MODEL_PATH=(path do .gguf). Python via venv — evita pip antigo do Ubuntu e PEP 668.
#
# Build (linux/amd64 para Runpod):
#   docker build --no-cache --platform linux/amd64 -f Dockerfile -t kiaia-server:cuda .
#
# Variáveis no Runpod (Environment):
#   MODEL_PATH, LLAMA_PORT, LLAMA_CTX, N_GPU_LAYERS, DEFAULT_*

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# Wheels com pip do venv (recente); não usa ``apt install python3-pip``.
FROM ubuntu:24.04 AS pydeps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && python3 -m venv /opt/wbuild \
    && /opt/wbuild/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /opt/wbuild/bin/pip wheel --no-cache-dir --wheel-dir /wheels runpod httpx \
    && rm -rf /root/.cache/pip \
    && rm -rf /var/lib/apt/lists/*

FROM ${LLAMA_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
# Volume Runpod: montar em /models para este default coincidir com o nome do .gguf no disco.
# Se o mount for outro (ex. /workspace), define MODEL_PATH nas Environment Variables do endpoint.
ENV MODEL_PATH=/models/Qwen3-8B-F16-Q4_K_M.gguf
ENV LLAMA_PORT=8080
ENV LLAMA_CTX=8192
ENV N_GPU_LAYERS=99
ENV DEFAULT_MAX_NEW_TOKENS=2048
ENV DEFAULT_TEMPERATURE=0.8
ENV DEFAULT_TOP_P=0.9

COPY --from=pydeps /wheels /wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir --no-index --find-links=/wheels runpod httpx \
    && rm -rf /wheels /root/.cache/pip \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

COPY handler.py /app/handler.py

ENTRYPOINT []

CMD ["python", "-u", "/app/handler.py"]
