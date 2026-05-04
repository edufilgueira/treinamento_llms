# Runpod Serverless: Python chama llama-server (CUDA) uma vez por worker; handler = HTTP local.
# Base: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# Build (linux/amd64 para Runpod):
#   docker build --platform linux/amd64 -f Dockerfile -t SEU_USER/llama-qwen-runpod:latest .
#   docker push SEU_USER/llama-qwen-runpod:latest
#
# Variáveis no Runpod (Environment):
#   MODEL_PATH, LLAMA_PORT, LLAMA_CTX, N_GPU_LAYERS
#   DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P — espelhar app_global do admin Oráculo (não há ligação automática à BD).

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

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages runpod httpx

COPY handler.py /app/handler.py
COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

# A imagem base (llama.cpp:server-cuda) define ENTRYPOINT = llama-server; sem isto o CMD
# passa a ser argumentos do binário → "error: invalid argument: python3".
ENTRYPOINT []

# -u: logs sem buffer (melhor no Runpod).
CMD ["python3", "-u", "/app/handler.py"]
