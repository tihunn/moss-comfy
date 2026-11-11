"""ComfyUI node: text -> sound effect with MOSS-SoundEffect v2, run in its own virtual environment."""

import torch

import comfy.model_management as mm
import comfy.utils

from .bridge import Cancelled, get_bridge

MAX_SEED = 0xFFFFFFFF


class MossSoundEffect:
    """One prompt, several variants (seed, seed+1, ...): a batch of AUDIO, one file per variant on save."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "heavy wooden door creaks open slowly, old hinges, close"}),
                "seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 30.0, "step": 0.1,
                                      "tooltip": "Length of the sound. Generation time does not depend on it: the model always denoises 30 s."}),
                "variants": ("INT", {"default": 4, "min": 1, "max": 32,
                                     "tooltip": "How many takes: seeds seed, seed+1, ... Each take can be repeated with its own seed and variants=1."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 500}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "keep_loaded_minutes": ("INT", {"default": 10, "min": 0, "max": 1440,
                                                "tooltip": "The MOSS process keeps the model in VRAM this long after the last run, then exits. 0 - exit right away."}),
                "unload_comfy_models": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Unload ComfyUI's own models from VRAM first: MOSS runs in another process and ComfyUI does not see its memory."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS"
    DESCRIPTION = "MOSS-SoundEffect v2 (OpenMOSS, Apache-2.0): text to sound effect, 48 kHz, up to 30 s."

    def generate(self, prompt, seconds, variants, seed, steps, cfg, negative_prompt="", sigma_shift=5.0,
                 keep_loaded_minutes=10, unload_comfy_models=True):
        if unload_comfy_models:
            mm.unload_all_models()
            mm.soft_empty_cache()
        bridge = get_bridge()
        bar = comfy.utils.ProgressBar(variants * steps)
        waves, rate = [], None
        for i in range(variants):
            last = i == variants - 1
            try:
                data, channels, frames, rate = bridge.generate(
                    prompt=prompt, seconds=seconds, seed=(seed + i) % (MAX_SEED + 1), steps=steps, cfg=cfg,
                    sigma_shift=sigma_shift, negative_prompt=negative_prompt,
                    # do not let the worker quit between the takes of one run
                    idle_seconds=keep_loaded_minutes * 60 if last else max(keep_loaded_minutes * 60, 60),
                    on_progress=lambda step, total, i=i: bar.update_absolute(i * steps + step),
                    should_cancel=mm.processing_interrupted,
                )
            except Cancelled:
                mm.throw_exception_if_processing_interrupted()
                raise mm.InterruptProcessingException()
            waves.append(torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(channels, frames))
        return ({"waveform": torch.stack(waves), "sample_rate": rate},)


NODE_CLASS_MAPPINGS = {"MossSoundEffect": MossSoundEffect}
NODE_DISPLAY_NAME_MAPPINGS = {"MossSoundEffect": "MOSS Sound Effect"}
