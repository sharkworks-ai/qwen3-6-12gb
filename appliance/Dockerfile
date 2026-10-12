FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG GIT_COMMIT=unknown
# Exact versions so identical source and configuration build identical environments;
# these are the versions validated on 2x RTX 5090.
ARG LLAMA_CPP_REF=097cf33d9499b1eade67d51e16d7c75427e7b267
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
RUN python3 -m pip install --no-cache-dir --break-system-packages torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url ${PYTORCH_INDEX_URL}
RUN python3 -m pip install --no-cache-dir --break-system-packages \
  "transformers==5.19.0" "peft==0.21.2" "trl==1.15.0" "accelerate==1.15.0" \
  "bitsandbytes==0.50.2" "datasets==5.1.0" "safetensors==0.8.0" "torchao==0.18.0" \
  "sentencepiece==0.2.2" "protobuf==7.36.2" \
  "fastapi==0.143.0" "uvicorn[standard]==0.54.0" "jinja2==3.1.6" "python-multipart==0.0.32" "itsdangerous==2.2.0" \
  "huggingface_hub==2.2.0" "httpx==0.28.1" "PyYAML==6.0.3" "pydantic==2.14.1" "psutil==7.2.2" "rich==15.0.0" "typer==0.27.3"
# A commit, not a branch: fetch it directly (clone --branch only accepts names).
RUN git init -q /opt/llama.cpp && \
    git -C /opt/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp ${LLAMA_CPP_REF} && \
    git -C /opt/llama.cpp checkout -q FETCH_HEAD && \
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
# Reconstruction windows run near the 32 GB limit; expandable segments avoid
# fragmentation OOMs when the next window's quantizers are allocated.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV QWEN12G_GPU_BACKEND=cuda QWEN12G_DATA_ROOT=/data HF_HOME=/data/hf-cache TRANSFORMERS_CACHE=/data/hf-cache PYTHONUNBUFFERED=1 PYTHONPATH=/app TOKENIZERS_PARALLELISM=false
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python3","-m","appliance.app"]
