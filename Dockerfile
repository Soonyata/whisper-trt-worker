# whisper-trt-worker v2 — NGC-BASE architecture: NVIDIA's own TensorRT-LLM container is the base,
# so the entire torch/tensorrt/numpy/triton matrix is NVIDIA-certified, not hand-assembled.
#
# HARD RULE (learned 2026-08-09, G1 failure + diag pod 33ccjkxwx3ouby): NEVER install anything
# that can move the torch family. The base ships NVIDIA's custom torch (2.10.0a0+…nv25.12);
# the image's TRT-LLM bindings are ABI-locked to it. Installing torchaudio (or any package
# whose deps include torch) drags in a PyPI torch and breaks tensorrt_llm.bindings with
# undefined C++ symbols. Torch-adjacent add-ons go in with --no-deps; silero's unused
# torchaudio import is stubbed in the worker. Base already ships numpy 1.26.4 — no pin needed.
# This exact add-on set was certified on a pristine-base pod: TRTLLM_STILL_OK after installs.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.1
ENV DEBIAN_FRONTEND=noninteractive
ARG TRT_TAG=v1.2.1
ARG CKPT_URL=https://github.com/Soonyata/whisper-trt-worker/releases/download/v0.1.0/ckpt_turbo_int8.tgz

RUN apt-get update -qq && apt-get install -y -qq ffmpeg wget && rm -rf /var/lib/apt/lists/*

# Stable layers first (never change across worker iterations): vendored example + checkpoint.
RUN mkdir -p /opt/whisper-example/assets && cd /opt/whisper-example \
    && for F in run.py tokenizer.py whisper_utils.py; do \
         wget -q "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/${TRT_TAG}/examples/models/core/whisper/$F"; done \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken \
    && wget -q -P assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
    && wget -qO- "$CKPT_URL" | tar xz \
    && test -d ckpt_turbo_int8/encoder

# Add-ons, exactly as pod-certified: torch-adjacent packages with --no-deps, the rest plain.
RUN pip install --no-cache-dir --no-deps silero-vad openai-whisper \
    && pip install --no-cache-dir runpod soundfile kaldialign tiktoken more-itertools \
    && python3 -c "import torch, numpy; assert torch.__version__.startswith('2.10.0a0'), torch.__version__; assert numpy.__version__.startswith('1.'), numpy.__version__; print('ADDONS_OK torch', torch.__version__, 'numpy', numpy.__version__)"

COPY turbo_trt_worker.py /opt/turbo_trt_worker.py
COPY boot.sh /opt/boot.sh
RUN chmod +x /opt/boot.sh
CMD ["/opt/boot.sh"]
