# whisper-trt-worker — RunPod serverless worker: whisper large-v3-turbo on TensorRT-LLM engines.
# VAD-chunked (silero), int8 weight-only, batch-32, C++ inflight-batching runtime.
# Engines are GPU-arch-specific and are built ON WORKER BOOT (~30s) from the baked
# pre-converted TRT-LLM checkpoint (downloaded from this repo's release assets).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV DEBIAN_FRONTEND=noninteractive
ARG TRT_TAG=v1.2.1
ARG CKPT_BASE=https://github.com/Soonyata/whisper-trt-worker/releases/download/v0.1.0

RUN apt-get update -qq && apt-get install -y -qq libopenmpi3 openmpi-bin ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# TRT-LLM (cu13 wheel) + the unsuffixed cu13 runtime libs the cu12.8 base lacks.
# ORDER MATTERS: the torch/torchaudio/triton pin triple must come LAST — torchaudio pulls
# otherwise drag torch to a version whose ABI breaks TRT-LLM's compiled bindings.
RUN pip install --break-system-packages --no-cache-dir tensorrt_llm \
    && for P in nvidia-cublas nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink nvidia-cuda-nvcc; do \
         pip install --break-system-packages --no-cache-dir "$P" || true; done \
    && pip install --break-system-packages --no-cache-dir openai-whisper kaldialign soundfile silero-vad runpod \
    && pip install --break-system-packages --no-cache-dir "torch==2.9.*" "torchaudio==2.9.*" "triton==3.5.1"

# Vendored inference scripts from the NVIDIA TensorRT-LLM whisper example (Apache-2.0),
# pinned to the tag matching the installed wheel + whisper assets + the converted checkpoint.
RUN mkdir -p /opt/whisper-example/assets && cd /opt/whisper-example \
    && for F in run.py tokenizer.py whisper_utils.py; do \
         wget -q "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/${TRT_TAG}/examples/models/core/whisper/$F"; done \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
    && for P in aa ab ac ad ae; do wget -q "$CKPT_BASE/ckpt_part_$P"; done \
    && cat ckpt_part_* | tar xz && rm -f ckpt_part_* \
    && test -d ckpt_turbo_int8/encoder

COPY turbo_trt_worker.py /opt/turbo_trt_worker.py
COPY boot.sh /opt/boot.sh
RUN chmod +x /opt/boot.sh
CMD ["/opt/boot.sh"]
