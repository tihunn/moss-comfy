"""The ComfyUI node with stub comfy modules and the fake worker. Needs torch (any build, CPU is fine);
skipped without it.
"""

import importlib.util
import math
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


class Interrupt(BaseException):
    pass


def stub_comfy(state):
    mm = types.ModuleType("comfy.model_management")
    mm.InterruptProcessingException = Interrupt
    mm.unload_all_models = lambda: state.setdefault("calls", []).append("unload")
    mm.soft_empty_cache = lambda force=False: state.setdefault("calls", []).append("empty")
    mm.processing_interrupted = lambda: state.get("interrupt", False)

    def throw():
        if state.get("interrupt"):
            state["interrupt"] = False
            raise Interrupt()

    mm.throw_exception_if_processing_interrupted = throw

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            state["bar"] = self

        def update_absolute(self, value, total=None, preview=None):
            state.setdefault("progress", []).append(value)
            if state.get("interrupt_at") == value:
                state["interrupt"] = True

    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    comfy = types.ModuleType("comfy")
    comfy.model_management, comfy.utils = mm, utils
    sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils})


def load_package():
    for name in [n for n in sys.modules if n == "moss_comfy" or n.startswith("moss_comfy.")]:
        del sys.modules[name]  # fresh nodes.py bound to this test's stubs
    spec = importlib.util.spec_from_file_location("moss_comfy", os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["moss_comfy"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


@unittest.skipIf(torch is None, "torch is not installed")
class NodeTest(unittest.TestCase):
    def setUp(self):
        self.state = {}
        stub_comfy(self.state)
        pkg = load_package()
        self.bridge_mod = sys.modules["moss_comfy.bridge"]
        config = dict(self.bridge_mod.DEFAULTS, python=sys.executable,
                      env={"MOSS_SFX_FAKE": "1", "MOSS_SFX_FAKE_STEP_SEC": "0.01"})
        self.bridge_mod._bridge = self.bridge_mod.MossBridge(config, log=lambda t: None)
        self.node = pkg.NODE_CLASS_MAPPINGS["MossSoundEffect"]()

    def tearDown(self):
        self.bridge_mod._bridge.close()

    def test_variants_become_a_batch(self):
        (audio,) = self.node.generate("door creak", 0.5, variants=3, seed=5, steps=4, cfg=4.0)
        wave = audio["waveform"]
        self.assertEqual(tuple(wave.shape), (3, 1, 4000))
        self.assertEqual(wave.dtype, torch.float32)
        self.assertEqual(audio["sample_rate"], 8000)
        for i in range(3):  # take i has seed 5+i: fake sine of 205+i Hz
            self.assertAlmostEqual(wave[i, 0, 1].item(), 0.5 * math.sin(2 * math.pi * (205 + i) / 8000), places=6)
        self.assertEqual(self.state["progress"][-1], 12)
        self.assertEqual(self.state["calls"][:2], ["unload", "empty"])

    def test_no_unload_when_asked(self):
        self.node.generate("x", 0.2, variants=1, seed=0, steps=1, cfg=4.0, unload_comfy_models=False)
        self.assertNotIn("calls", self.state)

    def test_cancel_raises_comfy_interrupt(self):
        self.state["interrupt_at"] = 3
        with self.assertRaises(Interrupt):
            self.node.generate("x", 0.2, variants=2, seed=0, steps=50, cfg=4.0)
        self.assertFalse(self.state["interrupt"])  # ComfyUI's flag is reset for the next run
        self.assertIsNone(self.bridge_mod._bridge.proc)

    def test_input_types_cover_the_function(self):
        spec = self.node.INPUT_TYPES()
        names = set(spec["required"]) | set(spec["optional"])
        code = type(self.node).generate.__code__
        self.assertEqual(names, set(code.co_varnames[1:code.co_argcount]))


if __name__ == "__main__":
    unittest.main()
