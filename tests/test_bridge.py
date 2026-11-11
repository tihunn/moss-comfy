"""Bridge <-> worker over real pipes, with the fake model (MOSS_SFX_FAKE=1): no torch, no GPU.

    python -m unittest discover -s tests -v
"""

import math
import os
import struct
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bridge  # noqa: E402


def make(extra_env=None, **cfg):
    config = dict(bridge.DEFAULTS, python=sys.executable, env=dict({"MOSS_SFX_FAKE": "1"}, **(extra_env or {})))
    config.update(cfg)
    logs = []
    b = bridge.MossBridge(config, log=logs.append)
    return b, logs


def samples(data):
    return struct.unpack("<%df" % (len(data) // 4), data)


class BridgeTest(unittest.TestCase):
    def tearDown(self):
        if getattr(self, "b", None):
            self.b.close()

    def test_generate_returns_audio_of_the_asked_length(self):
        self.b, logs = make()
        data, channels, frames, rate = self.b.generate("dog bark", 1.5, seed=0, steps=5)
        self.assertEqual((channels, frames, rate), (1, 12000, 8000))
        self.assertEqual(len(data), 4 * frames)
        x = samples(data)
        self.assertAlmostEqual(x[1], 0.5 * math.sin(2 * math.pi * 200 / 8000), places=6)
        # print() inside the worker went to stderr, the protocol survived
        self.assertTrue(any("junk on stdout" in line for line in logs))

    def test_seed_changes_the_take_and_repeats_it(self):
        self.b, _ = make()
        a = self.b.generate("x", 0.5, seed=1, steps=1)[0]
        b = self.b.generate("x", 0.5, seed=2, steps=1)[0]
        c = self.b.generate("x", 0.5, seed=1, steps=1)[0]
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def test_progress_is_reported_every_step(self):
        self.b, _ = make()
        seen = []
        self.b.generate("x", 0.2, seed=0, steps=7, on_progress=lambda s, t: seen.append((s, t)))
        self.assertEqual(seen, [(i, 7) for i in range(1, 8)])

    def test_model_stays_loaded_between_requests(self):
        self.b, _ = make()
        self.b.generate("x", 0.2, seed=0, steps=1)
        pid = self.b.proc.pid
        self.b.generate("x", 0.2, seed=1, steps=1)
        self.assertEqual(self.b.proc.pid, pid)

    def test_unicode_prompt(self):
        self.b, _ = make()
        data, _, frames, _ = self.b.generate("скрип двери, «старые петли»", 0.2, seed=0, steps=1)
        self.assertEqual(len(data), 4 * frames)

    def test_bad_request_keeps_the_worker(self):
        self.b, _ = make()
        with self.assertRaisesRegex(bridge.WorkerError, "seconds must be"):
            self.b.generate("x", 31, seed=0, steps=1)
        pid = self.b.proc.pid
        self.b.generate("x", 0.2, seed=0, steps=1)
        self.assertEqual(self.b.proc.pid, pid)

    def test_cancel_kills_the_worker_and_the_next_run_restarts_it(self):
        self.b, _ = make({"MOSS_SFX_FAKE_STEP_SEC": "0.05"})
        started = threading.Event()
        with self.assertRaises(bridge.Cancelled):
            self.b.generate("x", 0.2, seed=0, steps=200, on_progress=lambda s, t: started.set(),
                            should_cancel=started.is_set)
        self.assertIsNone(self.b.proc)
        data, _, frames, _ = self.b.generate("x", 0.2, seed=0, steps=1)
        self.assertEqual(len(data), 4 * frames)

    def test_idle_zero_frees_the_process_right_after(self):
        self.b, _ = make()
        self.b.generate("x", 0.2, seed=0, steps=1, idle_seconds=0)
        self.assertIsNone(self.b.proc)
        self.b.generate("x", 0.2, seed=0, steps=1, idle_seconds=0)

    def test_idle_timer_quits_and_the_next_request_restarts(self):
        self.b, _ = make()
        self.b.generate("x", 0.2, seed=0, steps=1, idle_seconds=0.3)
        proc = self.b.proc
        proc.wait(timeout=10)  # the worker left on its own
        data, _, frames, _ = self.b.generate("x", 0.2, seed=0, steps=1)
        self.assertEqual(len(data), 4 * frames)
        self.assertIsNot(self.b.proc, proc)

    def test_load_failure_is_reported_with_the_reason(self):
        self.b, _ = make({"MOSS_SFX_FAKE_FAIL": "1"})
        with self.assertRaisesRegex(bridge.WorkerError, "fake model failed to load"):
            self.b.generate("x", 0.2, seed=0, steps=1)

    def test_missing_python_says_what_to_do(self):
        self.b, _ = make(python="")
        with self.assertRaisesRegex(bridge.WorkerError, "config.json"):
            self.b.generate("x", 0.2, seed=0, steps=1)
        self.b, _ = make(python=os.path.join(ROOT, "no", "python.exe"))
        with self.assertRaisesRegex(bridge.WorkerError, "not found"):
            self.b.generate("x", 0.2, seed=0, steps=1)

    def test_config_file_and_env(self):
        path = os.path.join(ROOT, "tests", "_tmp_config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"python": "C:\\\\a\\\\python.exe", "hf_home": "D:\\\\hf"}')
        old = os.environ.pop("MOSS_SFX_PYTHON", None)
        try:
            cfg = bridge.load_config(path)
            self.assertEqual((cfg["python"], cfg["hf_home"], cfg["device"]), ("C:\\a\\python.exe", "D:\\hf", "cuda"))
            os.environ["MOSS_SFX_PYTHON"] = "E:\\py.exe"
            self.assertEqual(bridge.load_config(path)["python"], "E:\\py.exe")
        finally:
            os.remove(path)
            os.environ.pop("MOSS_SFX_PYTHON", None)
            if old is not None:
                os.environ["MOSS_SFX_PYTHON"] = old


if __name__ == "__main__":
    unittest.main()
