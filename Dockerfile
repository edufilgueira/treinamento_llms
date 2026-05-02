# Imagem oficial ggml-org (sem compilar). Base escolhida por --build-arg LLAMA_BASE.
# https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
#
# GPU (Runpod, máquina com NVIDIA + nvidia-container-toolkit):
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=server-cuda -t llama-qwen-server:cuda .
#   docker run -d --name llama-qwen --gpus all -p 8080:8080 --restart unless-stopped llama-qwen-server:cuda
#
# CPU (VPS sem GPU — sem --gpus):
#   docker build --platform linux/amd64 -f Dockerfile --build-arg LLAMA_BASE=server -t llama-qwen-server:cpu .
#   docker run -d --name llama-qwen -p 8080:8080 --restart unless-stopped llama-qwen-server:cpu
#
# Omissão = CUDA (compatível com o fluxo anterior).
# Tags fixas no GHCR (opcional): server-cuda12-bXXXX, etc.

ARG LLAMA_BASE=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_BASE}

WORKDIR /models

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

EXPOSE 8080

CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "8192", "--reasoning", "off", "--reasoning-budget", "0"]
