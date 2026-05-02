# Duplicado de /Dockerfile na raiz — mesmo contexto de build (. na raiz do repo).
#
#   docker build -f docker/llama-cpp/Dockerfile -t llama-qwen-server:latest .

ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

ARG LLAMA_CPP_REF=master
ARG BUILD_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggerganov/llama.cpp.git . \
    || (git clone https://github.com/ggerganov/llama.cpp.git . \
        && git checkout "${LLAMA_CPP_REF}")

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

RUN cmake --build build -j "${BUILD_JOBS}"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/ /usr/local/bin/
RUN ldconfig

COPY tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf /models/model.gguf

ENV MODEL_PATH=/models/model.gguf
ENV LLAMA_PORT=8080
ENV LLAMA_CTX=8192

EXPOSE 8080

ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/llama-server -m \"$MODEL_PATH\" --host 0.0.0.0 --port \"${LLAMA_PORT}\" -c \"${LLAMA_CTX}\" --reasoning off --reasoning-budget 0"]
