"""The ComfyUI side of the bridge: finds the MOSS virtual environment, keeps one worker.py process
alive in it and sends it requests. Standard library only — no torch here, so the tests run anywhere.

Settings, later ones win:
    config.json next to this file   {"python": "...", "hf_home": "...", ...} (see config.example.json)
    environment                      MOSS_SFX_PYTHON, MOSS_SFX_HF_HOME, MOSS_SFX_MODEL
"""

import atexit
import collections
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "worker.py")
DEFAULTS = {
    "python": "",
    "hf_home": "",
    "model": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    "device": "cuda",
    "dtype": "bfloat16",
    "env": {},
}
ENV_KEYS = {"python": "MOSS_SFX_PYTHON", "hf_home": "MOSS_SFX_HF_HOME", "model": "MOSS_SFX_MODEL"}


class Cancelled(Exception):
    """The user pressed Cancel in ComfyUI; the worker was killed."""


class WorkerError(RuntimeError):
    pass


class WorkerIdle(Exception):
    """The worker quit on its idle timer just as a request arrived; start it again."""


def load_config(path=os.path.join(HERE, "config.json")):
    cfg = dict(DEFAULTS)
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            cfg.update(json.load(f))
    for key, var in ENV_KEYS.items():
        if os.environ.get(var):
            cfg[key] = os.environ[var]
    return cfg


class MossBridge:
    def __init__(self, config=None, log=None):
        self.config = config
        self.log = log or (lambda text: print("[MOSS] " + text, flush=True))
        self.proc = None
        self.messages = None
        self.stderr_tail = collections.deque(maxlen=40)
        self.ready = None
        self.next_id = 0
        self.lock = threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix="moss_sfx_")
        atexit.register(self.close)

    # --- process -----------------------------------------------------------------------

    def _python(self, cfg):
        python = cfg.get("python") or ""
        if not python:
            raise WorkerError(
                "MOSS python is not set. Copy config.example.json to config.json in %s and put the path "
                "to python.exe of the MOSS virtual environment into \"python\" (or set MOSS_SFX_PYTHON)." % HERE)
        if not os.path.isfile(python):
            raise WorkerError("MOSS python not found: %s (config.json, \"python\")" % python)
        return python

    def _start(self):
        cfg = self.config if self.config is not None else load_config()
        python = self._python(cfg)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
        env.setdefault("TORCHDYNAMO_DISABLE", "1")  # torch.compile needs Triton, which Windows lacks
        env["MOSS_SFX_MODEL"] = str(cfg.get("model") or DEFAULTS["model"])
        env["MOSS_SFX_DEVICE"] = str(cfg.get("device") or DEFAULTS["device"])
        env["MOSS_SFX_DTYPE"] = str(cfg.get("dtype") or DEFAULTS["dtype"])
        if cfg.get("hf_home"):
            env["HF_HOME"] = str(cfg["hf_home"])
        env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.log("starting worker: %s" % python)
        # -I: the worker's own site-packages only, nothing from ComfyUI's environment
        self.proc = subprocess.Popen(
            [python, "-I", "-u", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=self.tmp, creationflags=flags)
        self.messages = queue.Queue()
        self.stderr_tail.clear()
        threading.Thread(target=self._read_stdout, args=(self.proc, self.messages), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()

    def _read_stdout(self, proc, messages):
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except ValueError:
                self.log("worker: " + line)
        proc.stdout.close()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        messages.put({"type": "exit"})

    def _read_stderr(self, proc):
        for raw in proc.stderr:
            # download bars redraw with \r: keep the last state only
            line = raw.decode("utf-8", "replace").rstrip().split("\r")[-1]
            if line:
                self.stderr_tail.append(line)
                self.log(line)
        proc.stderr.close()

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        proc, self.proc, self.ready = self.proc, None, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()

    def close(self):
        """Ask the worker to quit, kill it if it does not."""
        proc, self.proc, self.ready = self.proc, None, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(b'{"type": "quit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _wait(self, should_cancel, accept):
        """Next message accepted by accept(); raises if the worker dies or the user cancels."""
        while True:
            if should_cancel():
                self.kill()
                raise Cancelled()
            try:
                msg = self.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = msg.get("type")
            if kind in ("exit", "bye") and not accept(msg):
                if self.proc is not None:
                    self.proc.wait()
                self.proc, self.ready = None, None
                reason = msg.get("reason", "crash")
                if reason == "idle":
                    raise WorkerIdle()
                tail = "\n".join(list(self.stderr_tail)[-15:])
                raise WorkerError("MOSS worker exited (%s). Last output:\n%s" % (reason, tail))
            if kind == "error" and msg.get("id") is None:
                self.kill()
                raise WorkerError(msg.get("message", "worker error") + "\n" + msg.get("trace", ""))
            if accept(msg):
                return msg

    def _ensure(self, should_cancel):
        if self._alive() and self.ready:
            return
        self.kill()
        self._start()
        self.ready = self._wait(should_cancel, lambda m: m.get("type") == "ready")
        self.log("model ready: %s Hz, up to %s s" % (self.ready["sample_rate"], self.ready["max_seconds"]))

    # --- requests ----------------------------------------------------------------------

    def generate(self, prompt, seconds, seed, steps=100, cfg=4.0, sigma_shift=5.0, negative_prompt="",
                 idle_seconds=600, on_progress=None, should_cancel=None):
        """One sound. Returns (raw float32 bytes [C, T], channels, frames, sample_rate)."""
        should_cancel = should_cancel or (lambda: False)
        req = {"type": "generate", "prompt": prompt, "negative_prompt": negative_prompt, "seconds": seconds,
               "seed": seed, "steps": steps, "cfg": cfg, "sigma_shift": sigma_shift, "idle_seconds": idle_seconds}
        with self.lock:
            try:
                return self._generate(req, on_progress, should_cancel)
            except WorkerIdle:
                return self._generate(req, on_progress, should_cancel)

    def _generate(self, req, on_progress, should_cancel):
        self._ensure(should_cancel)
        self.next_id += 1
        req_id = self.next_id
        req = dict(req, id=req_id, out=os.path.join(self.tmp, "%d.f32" % req_id))
        try:
            # ASCII JSON: the worker's stdin uses the system code page
            self.proc.stdin.write((json.dumps(req, ensure_ascii=True) + "\n").encode("ascii"))
            self.proc.stdin.flush()
        except OSError:
            self.kill()
            raise WorkerIdle()

        def mine(m):
            if m.get("id") != req_id:
                return False
            if m.get("type") == "progress":
                if on_progress:
                    on_progress(m["step"], m["total"])
                return False
            return True

        msg = self._wait(should_cancel, mine)
        if msg.get("type") == "error":
            raise WorkerError(msg.get("message", "worker error"))
        with open(msg["path"], "rb") as f:
            data = f.read()
        os.remove(msg["path"])
        if req["idle_seconds"] <= 0:
            self._wait(should_cancel, lambda m: m.get("type") in ("bye", "exit"))
            if self.proc is not None:
                self.proc.wait()
            self.proc, self.ready = None, None
        return data, msg["channels"], msg["frames"], msg["sample_rate"]


_bridge = None


def get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = MossBridge()
    return _bridge


if __name__ == "__main__":
    # Quick check from a console: python bridge.py "dog barking twice" 2
    bridge = MossBridge()
    data, ch, frames, sr = bridge.generate(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0, seed=0,
                                           on_progress=lambda s, t: print("\r%d/%d" % (s, t), end=""))
    print("\n%d channel(s), %d frames, %d Hz" % (ch, frames, sr))
    bridge.close()
