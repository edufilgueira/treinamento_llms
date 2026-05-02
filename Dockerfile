# llama.cpp + llama-server (GPU) com modelo GGUF embutido.
#
# Construir a partir da raiz do repositório (onde está tools/quantized_model/):
#   docker build -f docker/llama-cpp/Dockerfile -t llama-qwen-server:latest .
#
# Executar (requer NVIDIA Container Toolkit na máquina host):
#   docker run --gpus all -p 8080:8080 llama-qwen-server:latest
#
# Trocar commit/branch do llama.cpp:
#   docker build -f docker/llama-cpp/Dockerfile --build-arg LLAMA_CPP_REF=b5602 -t llama-qwen-server:latest .

ARG CUDA_VERSION=12.4.1

# --- build ---
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggerganov/llama.cpp.git . \
    || (git clone https://github.com/ggerganov/llama.cpp.git . \
        && git checkout "${LLAMA_CPP_REF}")

# Compilação sem GPU no build: cobre GPUs comuns em cloud (Runpod, etc.).
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real"

RUN cmake --build build -j "$(nproc)"

# --- runtime ---
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bibliotecas partilhadas (.so) costumam ficar ao lado do binário; copiar bin/ evita falhas de glob.
COPY --from=build /src/build/bin/ /usr/local/bin/
RUN ldconfig

# Modelo quantizado (contexto de build = raiz do     treinamento_llms)
COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/Qwen3-8B-F16-Q4_K_M.gguf.gguf

ENV MODEL_PATH=/models/Qwen3-8B-F16-Q4_K_M.gguf.gguf
ENV LLAMA_PORT=8080
ENV LLAMA_CTX=8192

EXPOSE 8080

# Mesmas flags do guia local: host 0.0.0.0, ctx 8192, reasoning desligado.
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/llama-server -m \"$MODEL_PATH\" --host 0.0.0.0 --port \"${LLAMA_PORT}\" -c \"${LLAMA_CTX}\" --reasoning off --reasoning-budget 0"]
