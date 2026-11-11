"""MOSS-SoundEffect worker: runs inside the MOSS virtual environment, not inside ComfyUI.

The ComfyUI node (bridge.py) starts this script with the venv's python and talks to it with one
JSON object per line: requests on stdin, replies on stdout. Everything else the libraries print
goes to stderr, so a stray print() can not break the protocol.

Replies:
    {"type": "ready", "sample_rate": 48000, "max_seconds": 30}
    {"type": "progress", "id": 1, "step": 3, "total": 100}
    {"type": "result", "id": 1, "path": "...", "channels": 1, "frames": 96000, "sample_rate": 48000}
    {"type": "error", "id": 1, "message": "...", "trace": "..."}
    {"type": "bye", "reason": "idle" | "eof" | "quit"}

The waveform goes through a file: raw little-endian float32, channel after channel ([C, T]).

The model stays loaded between requests; the worker exits when nothing came for idle_seconds
(the last request sets it) or when stdin closes (ComfyUI has exited) — the VRAM goes back.

MOSS_SFX_FAKE=1 replaces the model with a sine whose pitch depends on the seed: the bridge
tests run on any python with no torch and no GPU.
"""

import json
import math
import os
import queue
import struct
import sys
import threading
import time
import traceback

# The protocol owns the real stdout; anything else written to fd 1 lands in stderr.
_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FAKE = os.environ.get("MOSS_SFX_FAKE") == "1"
_send_lock = threading.Lock()


def send(msg):
    with _send_lock:
        _proto.write(json.dumps(msg, ensure_ascii=False) + "\n")
        _proto.flush()


def log(text):
    print("[moss-worker] " + text, file=sys.stderr, flush=True)


class FakePipeline:
    """Stand-in for MossSoundEffectPipeline: mono 8 kHz sine, frequency from the seed."""

    sample_rate = 8000
    max_inference_seconds = 30

    def __init__(self):
        if os.environ.get("MOSS_SFX_FAKE_FAIL") == "1":
            raise RuntimeError("fake model failed to load")
        print("junk on stdout must not reach the protocol")

    def generate(self, req, progress):
        steps = int(req["steps"])
        delay = float(os.environ.get("MOSS_SFX_FAKE_STEP_SEC", "0"))
        for i in range(steps):
            if delay:
                time.sleep(delay)
            progress(i + 1, steps)
        freq = 200 + int(req["seed"]) % 800
        frames = int(self.sample_rate * round(float(req["seconds"]), 1))
        data = [0.5 * math.sin(2 * math.pi * freq * n / self.sample_rate) for n in range(frames)]
        return struct.pack("<%df" % frames, *data), 1, frames


class MossPipeline:
    def __init__(self, model, device, dtype):
        import torch
        from moss_soundeffect_v2 import MossSoundEffectPipeline

        self.torch = torch
        log("loading %s on %s (%s)" % (model, device, dtype))
        t0 = time.time()
        self.pipe = MossSoundEffectPipeline.from_pretrained(model, torch_dtype=getattr(torch, dtype), device=device)
        self.sample_rate = self.pipe.sample_rate
        self.max_inference_seconds = self.pipe.max_inference_seconds
        log("loaded in %.1f s" % (time.time() - t0))

    def generate(self, req, progress):
        def progress_bar(iterable):
            items = list(iterable)
            for i, item in enumerate(items):
                yield item
                progress(i + 1, len(items))

        audio = self.pipe(
            prompt=req["prompt"],
            negative_prompt=req.get("negative_prompt", ""),
            seconds=float(req["seconds"]),
            num_inference_steps=int(req["steps"]),
            cfg_scale=float(req["cfg"]),
            sigma_shift=float(req.get("sigma_shift", 5.0)),
            seed=int(req["seed"]),
            progress_bar_cmd=progress_bar,
        )
        wav = audio[0].detach().to(self.torch.float32).cpu().contiguous()  # [C, T]
        channels, frames = int(wav.shape[0]), int(wav.shape[1])
        return wav.numpy().astype("<f4").tobytes(), channels, frames


def read_stdin(q):
    for line in sys.stdin:
        line = line.strip()
        if line:
            q.put(line)
    q.put(None)


def main():
    model = os.environ.get("MOSS_SFX_MODEL", "OpenMOSS-Team/MOSS-SoundEffect-v2.0")
    device = os.environ.get("MOSS_SFX_DEVICE", "cuda")
    dtype = os.environ.get("MOSS_SFX_DTYPE", "bfloat16")
    try:
        pipe = FakePipeline() if FAKE else MossPipeline(model, device, dtype)
    except BaseException as e:
        traceback.print_exc()
        send({"type": "error", "id": None, "message": "model load failed: %s" % e, "trace": traceback.format_exc()})
        return 1
    send({"type": "ready", "sample_rate": pipe.sample_rate, "max_seconds": pipe.max_inference_seconds})

    q = queue.Queue()
    threading.Thread(target=read_stdin, args=(q,), daemon=True).start()
    idle = 600.0
    while True:
        try:
            line = q.get(timeout=idle if idle > 0 else None)
        except queue.Empty:
            send({"type": "bye", "reason": "idle"})
            return 0
        if line is None:
            send({"type": "bye", "reason": "eof"})
            return 0
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            if req.get("type") == "quit":
                send({"type": "bye", "reason": "quit"})
                return 0
            if req.get("type") != "generate":
                raise ValueError("unknown request type %r" % req.get("type"))
            idle = float(req.get("idle_seconds", idle))
            seconds = round(float(req["seconds"]), 1)
            if not 0 < seconds <= pipe.max_inference_seconds:
                raise ValueError("seconds must be in (0, %d], got %s" % (pipe.max_inference_seconds, seconds))

            def progress(step, total):
                send({"type": "progress", "id": req_id, "step": step, "total": total})

            t0 = time.time()
            data, channels, frames = pipe.generate(req, progress)
            with open(req["out"], "wb") as f:
                f.write(data)
            log("seed %s: %.1f s of audio in %.1f s" % (req["seed"], frames / pipe.sample_rate, time.time() - t0))
            send({"type": "result", "id": req_id, "path": req["out"], "channels": channels,
                  "frames": frames, "sample_rate": pipe.sample_rate})
        except Exception as e:
            traceback.print_exc()
            send({"type": "error", "id": req_id, "message": str(e), "trace": traceback.format_exc()})
        if idle <= 0:
            send({"type": "bye", "reason": "idle"})
            return 0


if __name__ == "__main__":
    sys.exit(main())
