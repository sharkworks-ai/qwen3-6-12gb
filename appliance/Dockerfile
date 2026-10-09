FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG GIT_COMMIT=unknown
ARG LLAMA_CPP_REF=master
ARG LLAMA_CUDA_ARCHITECTURES="75;86;89;120"
ARG BUILD_JOBS=2
ARG GSQ_REF=03fc16484c369e3127225615d5e03e8d3a6043e3
ARG CATQ_REF=5a8fcd4f7e0366554b732d300a37e6ea467c3c35
LABEL org.opencontainers.image.source="https://github.com/sharkworks-ai/qwen3-6-12gb" org.opencontainers.image.revision=$GIT_COMMIT
# Optional extra root CAs (for example a TLS-inspecting proxy); see docker/extra-ca/.
COPY docker/extra-ca/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-dev git git-lfs curl ca-certificates build-essential cmake ninja-build pkg-config openssh-client rsync tini && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}
RUN python3 -m pip install --no-cache-dir --break-system-packages \
  "transformers>=5.17.0" "peft>=0.18.0" "trl>=1.14.0" "accelerate>=1.10.0" \
  "bitsandbytes>=0.48.0" "datasets>=4.0.0" "safetensors>=0.5.0" "torchao>=0.17.0" \
  "sentencepiece>=0.2" "protobuf>=5" \
  "fastapi>=0.115" "uvicorn[standard]>=0.34" "jinja2>=3.1" "python-multipart>=0.0.18" "itsdangerous>=2.2" \
  "huggingface_hub>=0.27" "httpx>=0.28" "PyYAML>=6.0" "pydantic>=2.9" "psutil>=6.0" "rich>=13.9" "typer>=0.12"
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES}" -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/compat -Wl,-rpath-link,/usr/local/cuda/compat' && \
    cmake --build /opt/llama.cpp/build --config Release -j${BUILD_JOBS} --target llama-quantize llama-imatrix llama-cli llama-server
WORKDIR /app
RUN git clone https://github.com/IST-DASLab/GSQ /opt/GSQ && git -C /opt/GSQ checkout --detach ${GSQ_REF} && \
    git clone https://github.com/IntelChina-AI/BitTern /opt/BitTern && git -C /opt/BitTern checkout --detach ${CATQ_REF} && \
    python3 -m pip install --no-cache-dir --break-system-packages lion-pytorch==0.2.5
COPY . /app
RUN python3 -m pip install --no-cache-dir --break-system-packages -e .
RUN useradd --create-home --uid 10001 appliance && mkdir -p /data && chown -R appliance:appliance /data /app /opt/GSQ /opt/BitTern
USER appliance
ENV QWEN12G_GPU_BACKEND=cuda QWEN12G_DATA_ROOT=/data HF_HOME=/data/hf-cache TRANSFORMERS_CACHE=/data/hf-cache PYTHONUNBUFFERED=1 PYTHONPATH=/app TOKENIZERS_PARALLELISM=false
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python3","-m","appliance.app"]
