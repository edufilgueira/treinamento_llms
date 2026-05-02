# Imagem oficial ggml-org (sem compilar). Base = --build-arg LLAMA_BASE
# https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
# https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
#
# GPU (Runpod, host com NVIDIA + nvidia-container-toolkit):
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=server-cuda -t llama-qwen-server:cuda .
#   # Se aparecer erro de .so em falta, usa a imagem completa:
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=full-cuda -t llama-qwen-server:cuda .
#   docker run -d --name llama-qwen --gpus all -p 8080:8080 --restart unless-stopped llama-qwen-server:cuda
#
# CPU (VPS sem GPU): preferir `full` — a tag `server` às vezes falha com libllama-common.so no GHCR.
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=full -t llama-qwen-server:cpu .
#   docker rm -f llama-qwen
#   docker run -d --name llama-qwen -p 8080:8080 --restart unless-stopped llama-qwen-server:cpu
#
# Omissão = server-cuda (GPU).

ARG LLAMA_BASE=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_BASE}

# `full` / `full-cuda` usam ENTRYPOINT /app/tools.sh; `server` já usa llama-server.
# Definir sempre llama-server evita tools.sh e alinha com a doc (ex.: --entrypoint /app/llama-cli para light).
ENTRYPOINT ["/app/llama-server"]

WORKDIR /app
ENV LD_LIBRARY_PATH=/app:/app/lib:${LD_LIBRARY_PATH}

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

EXPOSE 8080

CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "8192", "--reasoning", "off", "--reasoning-budget", "0"]
