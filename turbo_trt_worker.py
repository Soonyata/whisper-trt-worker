#!/usr/bin/env python3
"""whisper-trt production worker — VAD-chunked TRT-LLM turbo engines → canonical segments.

THE Stage-B artifact (trt/PLAN.md). Two entry points:
  · bench/pod mode:  python3 turbo_trt_worker.py <audio-file-or-url> [out.json]
  · serverless mode: RUNPOD_SERVERLESS=1 python3 turbo_trt_worker.py   (runpod handler loop)

Contract (identical to serverless/turbo_worker.py lean CT2 worker):
  in : {"audio_url": <url>, "guid": ..., "title": ...}
  out: {"guid","title","duration","language","segments":[{start,end,speaker:None,text}],"model"}

Pipeline: silero-VAD speech spans → greedy-pack into ≤30s chunks (REAL start offsets = segment
timestamps; silence skipped entirely — faster AND no hallucination fodder) → GPU log-mels →
WhisperTRTLLM.process_batch (batch 16, greedy, C++ IFB) → special-token strip → canonical segments.

Env: ENGINE_DIR (default /engines/eng_turbo_int8) · EXAMPLE_DIR (default /opt/whisper-example —
must hold run.py/whisper_utils.py/assets from TRT-LLM tag matching the installed wheel; see
build_turbo_engines.sh) · TRT_BATCH (16) · TRT_MAX_NEW_TOKENS (190; gate-3 watches truncation).

Gate-A numbers this productionizes: 9.7s / 92-min ep on a 3090 (568x RT, $0.00039/audio-hr),
word-identical spot-checks vs CT2. STAGE-B VALIDATED 2026-08-08: 10.2s/543x (b32 engines 9.8s/563x),
words +0.06% vs CT2, 10-episode diversity soak PASSED (PLAN.md).

CALLER GOTCHA: the MPI runtime slurps stdin — shell while-read loops invoking this worker MUST
redirect `< /dev/null` on the python call or the loop dies after one iteration.
"""
import json
import os
import re
import sys
import time
import types
import urllib.request

# The NGC release image ships no torchaudio, and installing one replaces NVIDIA's custom
# torch build (ABI break — G1 failure 2026-08-09). silero_vad imports torchaudio at module
# level but touches it only inside read_audio/save_audio, which this worker never calls
# (audio is decoded via ffmpeg+soundfile). A stub satisfies the import.
try:
    import torchaudio  # noqa: F401
except ImportError:
    _ta = types.ModuleType("torchaudio")
    _ta.__version__ = "0.0.0+stub"
    sys.modules["torchaudio"] = _ta

EXAMPLE_DIR = os.environ.get("EXAMPLE_DIR", "/opt/whisper-example")
ENGINE_DIR = os.environ.get("ENGINE_DIR", "/engines/eng_turbo_int8")
BATCH = int(os.environ.get("TRT_BATCH", "16"))
MAX_NEW = int(os.environ.get("TRT_MAX_NEW_TOKENS", "190"))
SR = 16000
CHUNK_S = 30.0


def _prefix(language="en"):
    """Per-request language → whisper task-token prefix (G0 flag: was hardcoded en)."""
    lang = re.sub(r"[^a-z]", "", (language or "en").strip().lower()) or "en"
    return f"<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"

sys.path.insert(0, EXAMPLE_DIR)

_MODEL = None
_VAD = None


def _ensure_engines():
    """Lazy per-arch engine build (first request pays ~30-60s ONCE per worker; the platform
    waits inside a claimed job by contract — unlike boot time, which serverless kills)."""
    import subprocess, time as _t
    import torch
    arch = "sm%d%d" % torch.cuda.get_device_capability()
    base = os.environ.get("ENGINE_BASE", "/engines")
    eng = os.path.join(base, arch)
    cache = os.environ.get("NV_ENGINE_CACHE")
    if cache and os.path.exists(os.path.join(cache, arch, "encoder", "rank0.engine")):
        return os.path.join(cache, arch)
    if not os.path.exists(os.path.join(eng, "encoder", "rank0.engine")):
        ckpt = os.environ.get("CKPT_DIR", "/opt/whisper-example/ckpt_turbo_int8")
        t0 = _t.time()
        os.makedirs(eng, exist_ok=True)
        def _build(tag, args):
            r = subprocess.run(["trtllm-build"] + args, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "")[-1500:]
                print(f"TRTLLM_BUILD_FAIL {tag}: {tail}", flush=True)
                raise RuntimeError(f"{tag} engine build failed: ...{tail[-400:]}")
        _build("encoder", ["--checkpoint_dir", f"{ckpt}/encoder",
                           "--output_dir", f"{eng}/encoder", "--moe_plugin", "disable",
                           "--max_batch_size", "32", "--gemm_plugin", "disable",
                           "--bert_attention_plugin", "float16",
                           "--max_input_len", "3000", "--max_seq_len=3000"])
        _build("decoder", ["--checkpoint_dir", f"{ckpt}/decoder",
                           "--output_dir", f"{eng}/decoder", "--moe_plugin", "disable",
                           "--max_beam_width", "4", "--max_batch_size", "32",
                           "--max_seq_len", "200", "--max_input_len", "14",
                           "--max_encoder_input_len", "3000", "--gemm_plugin", "float16",
                           "--bert_attention_plugin", "float16", "--gpt_attention_plugin", "float16"])
        print(f"ENGINE_BUILD_METRICS arch={arch} build_s={_t.time()-t0:.0f}", flush=True)
        if cache:
            import shutil
            os.makedirs(cache, exist_ok=True)
            shutil.copytree(eng, os.path.join(cache, arch), dirs_exist_ok=True)
    return eng


def _model():
    global _MODEL
    if _MODEL is None:
        eng = _ensure_engines()
        from run import WhisperTRTLLM                      # TRT-LLM example (version-matched)
        _MODEL = WhisperTRTLLM(eng, False, os.path.join(EXAMPLE_DIR, "assets"),
                               batch_size=BATCH, use_py_session=False, num_beams=1)
    return _MODEL


def _vad():
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    return _VAD


def _speech_chunks(wave):
    """silero speech spans → greedy-packed chunks of ≤30s with real start offsets (seconds)."""
    import torch
    from silero_vad import get_speech_timestamps
    spans = get_speech_timestamps(torch.from_numpy(wave), _vad(), sampling_rate=SR,
                                  min_silence_duration_ms=500, speech_pad_ms=200)
    chunks = []                                            # (start_sample, end_sample)
    for sp in spans:
        s, e = sp["start"], sp["end"]
        while e - s > int(CHUNK_S * SR):                   # hard-split monologue spans >30s
            chunks.append((s, s + int(CHUNK_S * SR)))
            s += int(CHUNK_S * SR)
        if chunks and (e - chunks[-1][0]) <= int(CHUNK_S * SR):
            chunks[-1] = (chunks[-1][0], e)                # pack into the open chunk
        else:
            chunks.append((s, e))
    return [(s / SR, e / SR, wave[s:e]) for s, e in chunks if e > s]


def _strip(text):
    return re.sub(r"<\|[^|]*\|>", "", str(text)).strip()


def transcribe_core(audio_path, language="en"):
    """audio file → canonical segments + timing report. The single testable unit."""
    import soundfile as sf
    import torch
    from whisper_utils import log_mel_spectrogram

    wave, sr = sf.read(audio_path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    assert sr == SR, f"expected 16k mono wav-decodable input, got sr={sr}"
    dur = len(wave) / SR

    model = _model()
    chunks = _speech_chunks(wave)
    t0 = time.time()
    segs, capped = [], 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        mels = []
        for (_s, _e, w) in batch:
            pad = max(0, int(CHUNK_S * SR) - len(w))
            m = log_mel_spectrogram(torch.from_numpy(w), model.n_mels, padding=pad,
                                    device="cuda", mel_filters_dir=os.path.join(EXAMPLE_DIR, "assets"))
            mels.append(m.unsqueeze(0))
        lens = torch.tensor([m.shape[2] for m in mels], dtype=torch.int32, device="cuda")
        outs = model.process_batch(mels, lens, _prefix(language), num_beams=1, max_new_tokens=MAX_NEW)
        for (s, e, _w), t in zip(batch, outs):
            txt = _strip(t[0] if isinstance(t, list) else t)
            if txt:
                if len(txt.split()) >= MAX_NEW * 0.72:     # crude truncation flag (gate 3)
                    capped += 1
                segs.append({"start": round(s, 2), "end": round(e, 2), "speaker": None, "text": txt})
    t_trans = time.time() - t0
    return {"duration": round(dur, 1), "language": "en", "segments": segs,
            "model": "large-v3-turbo+trt-int8",
            "_timing": {"transcribe_s": round(t_trans, 1), "rt_factor": round(dur / t_trans, 1),
                        "chunks": len(chunks), "cap_flagged": capped}}


def _fetch(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (podcast-transcriber)"})
    with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as f:
        while (b := r.read(1 << 20)):
            f.write(b)


def _to_wav(src, dst):
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", str(SR), "-ac", "1",
                    "-c:a", "pcm_s16le", dst], check=True, timeout=1800)


def handler(job):
    """RunPod serverless handler — same contract as the lean CT2 worker."""
    inp = job["input"]
    guid = inp["guid"]
    raw = f"/tmp/ep_{abs(hash(guid)) % 10**12}"
    try:
        _fetch(inp["audio_url"], raw + ".audio")
        _to_wav(raw + ".audio", raw + ".wav")
        rec = transcribe_core(raw + ".wav", language=inp.get("language", "en"))
        rec.pop("_timing", None)
        return {"guid": guid, "title": inp.get("title"), **rec}
    finally:
        for ext in (".audio", ".wav"):
            try:
                os.remove(raw + ext)
            except OSError:
                pass


if __name__ == "__main__":
    if os.environ.get("RUNPOD_SERVERLESS"):
        import runpod
        runpod.serverless.start({"handler": handler})
    else:
        src = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/trt_out.json"
        local = "/tmp/bench_in"
        if src.startswith("http"):
            _fetch(src, local + ".audio"); _to_wav(local + ".audio", local + ".wav")
            src = local + ".wav"
        elif not src.endswith(".wav"):
            _to_wav(src, local + ".wav"); src = local + ".wav"
        rec = transcribe_core(src)
        json.dump(rec, open(out, "w"))
        t = rec["_timing"]
        print(f"TRT_WORKER_OK {t['transcribe_s']}s · {t['rt_factor']}x RT · {t['chunks']} chunks · "
              f"{len(rec['segments'])} segs · {sum(len(s['text'].split()) for s in rec['segments']):,} words · "
              f"cap_flagged={t['cap_flagged']} → {out}")
