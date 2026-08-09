#!/usr/bin/env python3
"""Pod-batch harness — the 568× lane. Feed an episodes.jsonl to the TRT worker on a rented pod;
a prefetch thread downloads+decodes ahead so the GPU never waits on the network.

  python3 batch_pod.py episodes.jsonl [--shard I/N] [--out out] [--ahead 2] [--language en] < /dev/null

episodes.jsonl rows: {"guid": ..., "audio_url": ..., "title": ...}  (extra fields ignored)
Multi-pod: launch N pods with --shard 0/N .. N-1/N — each takes every Nth episode, no coordination.
Resume: ledger.jsonl in --out records every attempt; "ok" guids are skipped on restart.
MPI GOTCHA (inherited from TRT-LLM): ALWAYS invoke with stdin redirected (< /dev/null) —
the MPI runtime slurps stdin and kills shell while-read callers.

Env (same as the worker): EXAMPLE_DIR, CKPT_DIR, ENGINE_BASE, TRT_BATCH; PYTHONPATH must
include EXAMPLE_DIR (for run.py/whisper_utils.py).
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turbo_trt_worker as W  # noqa: E402  — the worker IS the engine room; this file only feeds it


def led(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("LEDGER", json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--shard", default="0/1", help="I/N — this pod takes episodes where idx %% N == I")
    ap.add_argument("--out", default="out")
    ap.add_argument("--ahead", type=int, default=2, help="prefetch depth (episodes decoded ahead)")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))

    os.makedirs(a.out, exist_ok=True)
    wavdir = os.path.join(a.out, "_wav")
    os.makedirs(wavdir, exist_ok=True)
    ledger = os.path.join(a.out, "ledger.jsonl")

    eps = [json.loads(l) for l in open(a.episodes) if l.strip()]
    eps = [e for k, e in enumerate(eps) if k % shard_n == shard_i]
    done = set()
    if os.path.exists(ledger):
        for l in open(ledger):
            try:
                r = json.loads(l)
                if r.get("status") == "ok":
                    done.add(r["guid"])
            except json.JSONDecodeError:
                pass
    todo = [e for e in eps if e["guid"] not in done]
    print(f"BATCH_POD shard={a.shard} episodes={len(eps)} done={len(done)} todo={len(todo)}", flush=True)

    # Prefetch pipeline: fetch + decode + VAD all happen on CPU worker threads, so the GPU
    # only ever sees ready-chunked audio. Queue holds (ep, dur, chunks, fetch_s, vad_s);
    # chunks carry the speech audio in RAM (~250-350MB per 100-min episode) — keep --ahead small.
    import concurrent.futures as _cf
    import soundfile as sf

    q = queue.Queue(maxsize=max(1, a.ahead))

    # silero's torch-JIT model is NOT safe for concurrent init/forward across threads
    # (v2 validation segfaulted, exit 139): warm it once on the main thread, then
    # serialize VAD behind a lock. Fetch+decode still overlap; VAD ~25s/ep serialized
    # still feeds a ~12s/ep GPU at ~240x sustained.
    W._vad()
    vad_lock = threading.Lock()

    def prep(ep):
        wav = os.path.join(wavdir, ep["guid"] + ".wav")
        t0 = time.time()
        src = wav + ".src"
        W._fetch(ep["audio_url"], src)
        W._to_wav(src, wav)
        os.remove(src)
        fetch_s = round(time.time() - t0, 1)
        t1 = time.time()
        wave, sr = sf.read(wav, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        assert sr == 16000, f"bad sr {sr}"
        dur = len(wave) / sr
        with vad_lock:
            chunks = W._speech_chunks(wave)
        os.remove(wav)
        return ep, dur, chunks, fetch_s, round(time.time() - t1, 1)

    def producer():
        print("PRODUCER_START", flush=True)
        with _cf.ThreadPoolExecutor(max_workers=max(1, a.ahead)) as ex_pool:
            futs = {ex_pool.submit(prep, ep): ep for ep in todo}
            for fut in _cf.as_completed(futs):
                ep = futs[fut]
                try:
                    q.put(fut.result())
                except Exception as ex:  # noqa: BLE001 — one bad episode must not sink the batch
                    led(ledger, {"guid": ep["guid"], "status": "fetch_error", "error": str(ex)[:300]})
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    t_start = time.time()
    n_ok = n_err = 0
    audio_s = 0.0
    while True:
        item = q.get()
        if item is None:
            break
        ep, dur, chunks, fetch_s, vad_s = item
        try:
            rec = W.transcribe_core(None, language=a.language, pre=(dur, chunks))
            timing = rec.pop("_timing", {})
            out_rec = {"guid": ep["guid"], "title": ep.get("title"), **rec}
            with open(os.path.join(a.out, ep["guid"] + ".json"), "w") as f:
                json.dump(out_rec, f)
            words = sum(len(s["text"].split()) for s in rec["segments"])
            audio_s += rec["duration"]
            n_ok += 1
            led(ledger, {"guid": ep["guid"], "status": "ok", "dur_s": rec["duration"],
                         "words": words, "segs": len(rec["segments"]), "fetch_s": fetch_s,
                         "vad_s": vad_s, "transcribe_s": timing.get("transcribe_s"),
                         "rt_x": timing.get("rt_factor"), "cap_flagged": timing.get("cap_flagged")})
        except Exception as ex:  # noqa: BLE001
            n_err += 1
            led(ledger, {"guid": ep["guid"], "status": "transcribe_error", "error": str(ex)[:300]})
    wall = time.time() - t_start
    print(f"BATCH_DONE ok={n_ok} err={n_err} audio_hrs={audio_s/3600:.2f} wall_s={wall:.0f} "
          f"sustained_rt={audio_s/max(wall,1):.0f}x", flush=True)


if __name__ == "__main__":
    main()
