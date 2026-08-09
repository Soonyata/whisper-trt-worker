# whisper-trt-worker v2 — NGC-BASE architecture: NVIDIA's own TensorRT-LLM container is the base,
# so the entire torch/tensorrt/numpy/triton matrix is NVIDIA-certified, not hand-assembled.
# We add ONLY: ffmpeg, the runpod SDK + VAD + audio IO, the vendored example scripts (version-matched
# tag), the pre-converted checkpoint, and our worker. Engines still build per-arch on worker boot.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.1
ENV DEBIAN_FRONTEND=noninteractive
ARG TRT_TAG=v1.2.1
ARG CKPT_URL=https://github.com/Soonyata/whisper-trt-worker/releases/download/v0.1.0/ckpt_turbo_int8.tgz

RUN apt-get update -qq && apt-get install -y -qq ffmpeg wget && rm -rf /var/lib/apt/lists/*

# Add-ons only — constrained so they cannot move the base's certified torch/numpy:
# torchaudio pinned to the base's torch minor; numpy<2 restated as a guard (upstream requirement).
RUN pip install --no-cache-dir "torchaudio==2.9.*" "numpy<2" runpod silero-vad soundfile openai-whisper kaldialign \
    && python3 -c "import numpy, torch, torchaudio; assert numpy.__version__.startswith('1.'), numpy.__version__; print('ADDONS_OK torch', torch.__version__, 'numpy', numpy.__version__)"

RUN mkdir -p /opt/whisper-example/assets && cd /opt/whisper-example \
    && for F in run.py tokenizer.py whisper_utils.py; do \
         wget -q "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/${TRT_TAG}/examples/models/core/whisper/$F"; done \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
    && wget -qO- "$CKPT_URL" | tar xz \
    && test -d ckpt_turbo_int8/encoder

COPY turbo_trt_worker.py /opt/turbo_trt_worker.py
COPY boot.sh /opt/boot.sh
RUN chmod +x /opt/boot.sh
CMD ["/opt/boot.sh"]
