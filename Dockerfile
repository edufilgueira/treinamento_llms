# llama-server com CUDA — imagem oficial (sem compilar localmente).
# https://github.com/ggerganov/llama.cpp/pkgs/container/llama.cpp
#
# Construir na raiz do repo (contexto com tools/quantized_model/):
#   docker build --platform linux/amd64 -f Dockerfile -t llama-qwen-server:latest .
#
# Runpod / GPU local:
#   docker run --gpus all -p 8080:8080 llama-qwen-server:latest
#
# Para fixar versão, troca a tag (ex. server-cuda12-b9002) no FROM — ver tags em GHCR.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

WORKDIR /models

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

# A imagem base já expõe llama-server em 8080 no HEALTHCHECK; mantemos 8080 (Oráculo usa 8080 por omissão).
EXPOSE 8080

CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "8192", "--reasoning", "off", "--reasoning-budget", "0"]
