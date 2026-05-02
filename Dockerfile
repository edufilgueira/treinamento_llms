# Imagem oficial ggml-org (sem compilar). Base = --build-arg LLAMA_BASE
# https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# GPU (Runpod, host com NVIDIA + nvidia-container-toolkit):
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=server-cuda -t llama-qwen-server:cuda .
#   # Se aparecer erro de .so em falta: --build-arg LLAMA_BASE=full-cuda
#   docker run -d --name llama-qwen --gpus all -p 8080:8080 --restart unless-stopped llama-qwen-server:cuda
#
# CPU (VPS sem GPU): preferir `full`.
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=full -t llama-qwen-server:cpu .
#   docker rm -f llama-qwen && docker run -d --name llama-qwen -p 8080:8080 --restart unless-stopped llama-qwen-server:cpu
#
# `full` / `full-cuda` usam tools.sh: o entrypoint repassa para `tools.sh --server` automaticamente.
# `server` / `server-cuda` chamam /app/llama-server direto.
#
# Omissão = server-cuda (GPU).

ARG LLAMA_BASE=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_BASE}

WORKDIR /app
ENV LD_LIBRARY_PATH=/app:/app/lib:${LD_LIBRARY_PATH}

COPY llama-server-entrypoint.sh /llama-server-entrypoint.sh
RUN chmod +x /llama-server-entrypoint.sh
ENTRYPOINT ["/llama-server-entrypoint.sh"]

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

EXPOSE 8080

CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "8192", "--reasoning", "off", "--reasoning-budget", "0"]
