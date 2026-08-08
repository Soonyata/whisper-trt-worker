# whisper-trt-worker

RunPod serverless worker: OpenAI **whisper large-v3-turbo** running on **TensorRT-LLM engines**
(int8 weight-only, batch-32, C++ inflight batching), with **silero-VAD chunking** producing
timestamped segments. Measured ~540–570× real-time on an RTX 3090 for long-form podcast audio.

- Engines are GPU-architecture-specific → built **on worker boot** (~30s) from the pre-converted
  TRT-LLM checkpoint shipped in this repo's release assets. `boot.sh` emits a `BOOT_METRICS` line.
- Request:  `{"input": {"audio_url": "...", "guid": "...", "title": "...", "language": "en"}}`
- Response: `{"guid","title","duration","language","segments":[{start,end,speaker,text}],"model"}`

Built on the [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) whisper example
(Apache-2.0), pinned to the tag matching the installed wheel. Whisper by OpenAI.
