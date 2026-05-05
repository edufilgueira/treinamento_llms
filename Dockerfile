# Runpod Serverless: handler em streaming (yield SSE do llama-server) + return_aggregate_stream;
# Oráculo consome GET .../stream/{job_id}. Base: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# Esta imagem NÃO inclui o .gguf (pull rápido). No Runpod: Network Volume montado em /models (ou outro path)
# e MODEL_PATH=/models/model.gguf (ou o ficheiro que tiveres no volume).
#
# Tamanho: ~2.x GB é sobretudo NVIDIA CUDA runtime + llama-server (ggml-org já usa alvo ``server``, o mais leve).
# Ganâncias aqui: wheels alinhados ao Ubuntu da base, remoção de pip após instalar, caches limpos.
#
# Build (linux/amd64 para Runpod):
#   docker build --platform linux/amd64 -f Dockerfile -t SEU_USER/llama-qwen-runpod:latest .
#   docker push SEU_USER/llama-qwen-runpod:latest
#
# Variáveis no Runpod (Environment):
#   MODEL_PATH, LLAMA_PORT, LLAMA_CTX, N_GPU_LAYERS
#   DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
#
# Opcional — imagem base CUDA13 (pode ser menor ou maior conforme versão); GPU Runpod tem de ser compatível.
#   docker build --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda13 ...
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# Mesma família que a base ggml (Ubuntu 24.04 + Python 3.12) → wheels compatíveis.
FROM ubuntu:24.04 AS pydeps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && python3 -m pip wheel --no-cache-dir --break-system-packages --wheel-dir /wheels runpod httpx \
    && rm -rf /root/.cache/pip \
    && apt-get purge -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

FROM ${LLAMA_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_PATH=/workspace/Qwen3-8B-F16-Q4_K_M.gguf
ENV LLAMA_PORT=8080
ENV LLAMA_CTX=8192
ENV N_GPU_LAYERS=99
ENV DEFAULT_MAX_NEW_TOKENS=2048
ENV DEFAULT_TEMPERATURE=0.8
ENV DEFAULT_TOP_P=0.9

COPY --from=pydeps /wheels /wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && python3 -m pip install --no-cache-dir --break-system-packages --no-index --find-links=/wheels runpod httpx \
    && rm -rf /wheels /root/.cache/pip \
    && apt-get purge -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY handler.py /app/handler.py

ENTRYPOINT []

CMD ["python3", "-u", "/app/handler.py"]
