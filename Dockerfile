# Imagem oficial ggml-org (sem compilar). Base = --build-arg LLAMA_BASE
# https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# GPU (Runpod, host com NVIDIA + nvidia-container-toolkit):
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=server-cuda -t llama-qwen-server:cuda .
#   # Se .so em falta: --build-arg LLAMA_BASE=full-cuda
#   docker run -d --name llama-qwen --gpus all -p 8080:8080 --restart unless-stopped llama-qwen-server:cuda
#
# CPU (VPS): `full` inclui tools.sh — o entrypoint injecta `--server` antes de `-m`.
#   docker build --no-cache --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=full -t llama-qwen-server:cpu .
#   docker rm -f llama-qwen && docker run -d --name llama-qwen -p 8080:8080 --restart unless-stopped llama-qwen-server:cpu
#
# Omissão = server-cuda (GPU).

ARG LLAMA_BASE=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_BASE}

WORKDIR /app
ENV LD_LIBRARY_PATH=/app:/app/lib:${LD_LIBRARY_PATH}

# Entrypoint embutido (LF garantidos). Imagem `full`: tools.sh exige 1º arg --server; imagem `server`: só llama-server.
RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -e' \
    'if [ -f /app/tools.sh ]; then exec /usr/bin/env bash /app/tools.sh --server "$@"; fi' \
    'exec /app/llama-server "$@"' \
    > /entry-llama.sh && chmod +x /entry-llama.sh

ENTRYPOINT ["/entry-llama.sh"]

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

EXPOSE 8080

CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "8192", "--reasoning", "off", "--reasoning-budget", "0"]
