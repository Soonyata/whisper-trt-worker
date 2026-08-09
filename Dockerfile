# whisper-trt-worker — RunPod serverless worker: whisper large-v3-turbo on TensorRT-LLM engines.
# VAD-chunked (silero), int8 weight-only, batch-32, C++ inflight-batching runtime.
# Engines are GPU-arch-specific and are built ON WORKER BOOT (~30s) from the baked
# pre-converted TRT-LLM checkpoint (downloaded from this repo's release assets).
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV DEBIAN_FRONTEND=noninteractive
ARG TRT_TAG=v1.2.1
ARG CKPT_URL=https://github.com/Soonyata/whisper-trt-worker/releases/download/v0.1.0/ckpt_turbo_int8.tgz

RUN apt-get update -qq && apt-get install -y -qq libopenmpi3 openmpi-bin ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# TRT-LLM (cu13 wheel) + the unsuffixed cu13 runtime libs the cu12.8 base lacks.
# ORDER MATTERS: the torch/torchaudio/triton pin triple must come LAST — torchaudio pulls
# otherwise drag torch to a version whose ABI breaks TRT-LLM's compiled bindings.
RUN pip install --break-system-packages --no-cache-dir tensorrt_llm \
    && for P in nvidia-cublas nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink nvidia-cuda-nvcc; do \
         pip install --break-system-packages --no-cache-dir "$P" || true; done \
    && pip install --break-system-packages --no-cache-dir --ignore-installed openai-whisper kaldialign soundfile silero-vad runpod \
    && pip uninstall --break-system-packages -y torchao torch-c-dlpack-ext torchvision || true
# SCORCHED-EARTH torch fix (pod-certified 2026-08-09): --ignore-installed above overlays torch 2.13
# files; force-reinstall of 2.9 leaves 2.13-ONLY ORPHANS (e.g. _inductor/kernel/custom_op.py) that
# dynamo's submodule walker imports -> circular-import crash. Physically remove the trees, then
# clean-install the pinned triple. Check imports only modules that EXIST in 2.9.
RUN pip uninstall --break-system-packages -y torch torchaudio triton pytorch-triton || true
RUN rm -rf /usr/local/lib/python3.12/dist-packages/torch \
           /usr/local/lib/python3.12/dist-packages/torchgen \
           /usr/local/lib/python3.12/dist-packages/functorch \
           /usr/local/lib/python3.12/dist-packages/torch-*.dist-info \
           /usr/local/lib/python3.12/dist-packages/torchaudio* \
           /usr/local/lib/python3.12/dist-packages/triton \
           /usr/local/lib/python3.12/dist-packages/triton-*.dist-info \
           /usr/local/lib/python3.12/dist-packages/pytorch_triton* \
    && pip install --break-system-packages --no-cache-dir "torch==2.9.*" "torchaudio==2.9.*" "torchvision==0.24.*" "triton==3.5.1" pillow "numpy<2" \
    && python3 -c "import torch; assert torch.__version__.startswith('2.9'), torch.__version__; import torch._inductor.lowering, torchvision, numpy; assert numpy.__version__.startswith('1.'), numpy.__version__; print('TORCH_CLEAN_OK', torch.__version__, torchvision.__version__, 'numpy', numpy.__version__)"

# Vendored inference scripts from the NVIDIA TensorRT-LLM whisper example (Apache-2.0),
# pinned to the tag matching the installed wheel + whisper assets + the converted checkpoint.
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
