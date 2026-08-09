#!/bin/bash
# whisper-trt worker boot v2 (lazy-engine design): start the runpod SDK IMMEDIATELY;
# engines are built inside the first request (turbo_trt_worker._ensure_engines) where the
# platform waits by contract. Boot-time work here is seconds, so FlashBoot/queue handshake
# happens instantly. (v1 built engines at boot — 30-90s pre-SDK — and serverless killed it.)
set -u
set -x
trap 'code=$?; echo "BOOT_EXIT code=$code"; sleep 120' EXIT
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export CKPT_DIR="${CKPT_DIR:-/opt/whisper-example/ckpt_turbo_int8}"
export EXAMPLE_DIR="${EXAMPLE_DIR:-/opt/whisper-example}"
export ENGINE_BASE="${ENGINE_BASE:-/engines}"
export TRT_BATCH="${TRT_BATCH:-32}" RUNPOD_SERVERLESS=1
echo "BOOT_METRICS mode=lazy sdk_start_immediate=1"
exec python3 /opt/turbo_trt_worker.py
