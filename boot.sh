#!/bin/bash
# whisper-trt worker boot: detect GPU arch -> ensure engines for THAT arch -> BOOT_METRICS -> handler.
# Engines are arch-compiled (SM86 Ampere / SM89 Ada); the image bakes the CONVERTED int8 checkpoint
# and builds engines here on first boot (~30s measured). G0 flag: NV_ENGINE_CACHE hook, DEFAULT OFF —
# set NV_ENGINE_CACHE=/runpod-volume/engines to reuse builds across same-arch workers (pins the
# endpoint to one datacenter; only adopt if BOOT_METRICS shows boot-storms matter).
set -u
# INSTRUMENTATION (temporary, staged for silent-crash diagnosis): trace every line to stdout,
# and on any exit hold the container 120s so worker logs can be read before the restart loop.
set -x
trap 'code=$?; echo "BOOT_EXIT code=$code"; sleep 120' EXIT
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
CKPT="${CKPT_DIR:-/opt/whisper-example/ckpt_turbo_int8}"
EX="${EXAMPLE_DIR:-/opt/whisper-example}"
ARCH=$(python3 -c "import torch; print('sm%d%d' % torch.cuda.get_device_capability())")
ENG="${ENGINE_BASE:-/engines}/$ARCH"
t0=$(date +%s); built=0
if [ -n "${NV_ENGINE_CACHE:-}" ] && [ -f "$NV_ENGINE_CACHE/$ARCH/encoder/rank0.engine" ]; then
  ENG="$NV_ENGINE_CACHE/$ARCH"
elif [ ! -f "$ENG/encoder/rank0.engine" ]; then
  mkdir -p "$ENG"
  trtllm-build --checkpoint_dir "$CKPT/encoder" --output_dir "$ENG/encoder" \
    --moe_plugin disable --max_batch_size 32 --gemm_plugin disable \
    --bert_attention_plugin float16 --max_input_len 3000 --max_seq_len=3000 > /tmp/enc_build.log 2>&1 \
    || { echo "BOOT_FAIL encoder build"; tail -5 /tmp/enc_build.log; exit 1; }
  trtllm-build --checkpoint_dir "$CKPT/decoder" --output_dir "$ENG/decoder" \
    --moe_plugin disable --max_beam_width 4 --max_batch_size 32 \
    --max_seq_len 200 --max_input_len 14 --max_encoder_input_len 3000 \
    --gemm_plugin float16 --bert_attention_plugin float16 --gpt_attention_plugin float16 > /tmp/dec_build.log 2>&1 \
    || { echo "BOOT_FAIL decoder build"; tail -5 /tmp/dec_build.log; exit 1; }
  built=1
  if [ -n "${NV_ENGINE_CACHE:-}" ]; then mkdir -p "$NV_ENGINE_CACHE" && cp -r "$ENG" "$NV_ENGINE_CACHE/" || true; fi
fi
echo "BOOT_METRICS arch=$ARCH build_s=$(( $(date +%s) - t0 )) built=$built engines=$ENG"
export ENGINE_DIR="$ENG" EXAMPLE_DIR="$EX" TRT_BATCH="${TRT_BATCH:-32}" RUNPOD_SERVERLESS=1
exec python3 /opt/turbo_trt_worker.py
