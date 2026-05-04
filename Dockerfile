# Runpod Serverless: handler em streaming (yield SSE do llama-server) + return_aggregate_stream;
# Oráculo consome GET .../stream/{job_id}. Base: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# Esta imagem NÃO inclui o .gguf (pull rápido). No Runpod: Network Volume montado em /models (ou outro path)
# e MODEL_PATH=/models/model.gguf (ou o ficheiro que tiveres no volume). Ver README / painel Runpod.
#
# Build (linux/amd64 para Runpod):
#   docker build --platform linux/amd64 -f Dockerfile -t SEU_USER/llama-qwen-runpod:latest .
#   docker push SEU_USER/llama-qwen-runpod:latest
#
# Variáveis no Runpod (Environment):
#   MODEL_PATH, LLAMA_PORT, LLAMA_CTX, N_GPU_LAYERS
#   DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P — espelhar app_global do admin Oráculo (não há ligação automática à BD).

# --- wheels fora da imagem final (menos ferramentas de compilação / cache pip visível no runtime)
FROM python:3.12-slim-bookworm AS pydeps
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels runpod httpx

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_PATH=/models/model.gguf
ENV LLAMA_PORT=8080
ENV LLAMA_CTX=8192
ENV N_GPU_LAYERS=99
# Omissões do handler quando o input Runpod não manda max_tokens / temperature / top_p (alinhado a server/db/auth_db omissões admin).
ENV DEFAULT_MAX_NEW_TOKENS=2048
ENV DEFAULT_TEMPERATURE=0.8
ENV DEFAULT_TOP_P=0.9

COPY --from=pydeps /wheels /wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip install --no-cache-dir --break-system-packages --no-index --find-links=/wheels runpod httpx \
    && rm -rf /wheels \
    && rm -rf /var/lib/apt/lists/*

COPY handler.py /app/handler.py

# A imagem base (llama.cpp:server-cuda) define ENTRYPOINT = llama-server; sem isto o CMD
# passa a ser argumentos do binário → "error: invalid argument: python3".
ENTRYPOINT []

# -u: logs sem buffer (melhor no Runpod).
CMD ["python3", "-u", "/app/handler.py"]
