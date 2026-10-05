#!/usr/bin/env bash
set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-120}"

mkdir -p /opt

if [[ ! -d /opt/llama.cpp/.git ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp
fi
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build /opt/llama.cpp/build -j --target \
  llama-server llama-bench llama-quantize llama-imatrix

if [[ ! -d /opt/llama.cpp-turboquant/.git ]]; then
  git clone --depth 1 -b feature/turboquant-kv-cache \
    https://github.com/TheTom/llama-cpp-turboquant /opt/llama.cpp-turboquant
fi
cmake -S /opt/llama.cpp-turboquant -B /opt/llama.cpp-turboquant/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build /opt/llama.cpp-turboquant/build -j --target \
  llama-server llama-bench llama-quantize llama-imatrix

if [[ ! -d /opt/BitTern/.git ]]; then
  git clone --depth 1 https://github.com/IntelChina-AI/BitTern /opt/BitTern
fi

echo "Quantization backends installed."
echo "BitTern/CAT-Q Qwen3.6 support remains experimental and must pass the adapter probe."
