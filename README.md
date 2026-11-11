# MOSS Sound Effect for ComfyUI

A ComfyUI node for [MOSS-SoundEffect v2](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(OpenMOSS, Apache-2.0): text → sound effect, 48 kHz, up to 30 s.

MOSS pins its own versions (`torch==2.9`, `transformers==4.57`, `numpy==1.26`) that clash with
ComfyUI's, so the model lives in **its own virtual environment**. The node starts a worker
process there and talks to it over pipes. Nothing is installed into ComfyUI's Python.

- One prompt → several takes (`seed`, `seed+1`, …) as one AUDIO batch: Save Audio writes a file per take.
- Progress bar and Cancel work; Cancel kills the worker.
- Before a run the node unloads ComfyUI's models (ComfyUI can not see the other process's VRAM).
- The worker keeps the model loaded for `keep_loaded_minutes` after the last run, then exits and
  frees VRAM. It also exits when ComfyUI exits.

## Install

**1. MOSS in its own venv** (Windows, Python 3.12; no conda needed):

```powershell
mkdir D:\comfy\moss; cd D:\comfy\moss
& "C:\path\to\Python312\python.exe" -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS\moss_soundeffect_v2
D:\comfy\moss\venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-cu128]"
```

Check: `D:\comfy\moss\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` → `True`.

**2. The node:**

```powershell
cd <ComfyUI>\custom_nodes
git clone <this repo> moss-comfy
copy moss-comfy\config.example.json moss-comfy\config.json
```

Edit `config.json`:

| Key | Meaning |
| --- | --- |
| `python` | `python.exe` of the MOSS venv — required |
| `hf_home` | where the model is downloaded (~ several GB on the first run); empty — the default HF cache |
| `model` | HF repo id or a local folder with the model |
| `device`, `dtype` | `cuda`, `bfloat16` |
| `env` | extra environment variables for the worker, e.g. `{"CUDA_VISIBLE_DEVICES": "0"}` |

Environment variables `MOSS_SFX_PYTHON`, `MOSS_SFX_HF_HOME`, `MOSS_SFX_MODEL` override the file.
`TORCHDYNAMO_DISABLE=1` is set for the worker: MOSS uses `torch.compile`, which needs Triton, and
Windows has none.

Restart ComfyUI. The node is **audio → MOSS → MOSS Sound Effect**. The first run downloads the model.

Check the worker without ComfyUI: `python bridge.py "dog barking twice" 2` from the node folder.

## Inputs

| Input | Default | |
| --- | --- | --- |
| `prompt` | | English or Chinese. Source, material, action, distance: `old wooden door creaks open slowly, close` |
| `seconds` | 3.0 | 0.1–30. Time to generate does not depend on it: the model always denoises 30 s and crops |
| `variants` | 4 | takes with seeds `seed`, `seed+1`, … — any take is repeatable with its seed and `variants = 1` |
| `seed` | 0 | |
| `steps`, `cfg` | 100, 4.0 | the authors' defaults |
| `negative_prompt` | empty | |
| `sigma_shift` | 5.0 | the authors' default |
| `keep_loaded_minutes` | 10 | 0 — the worker exits right after the run |
| `unload_comfy_models` | on | unload ComfyUI's models from VRAM before the run |

## From a script

Queue `examples/moss_sfx_api.json` with `POST /prompt` (`{"prompt": <that json>}`), wait on
`/history/<prompt_id>`, take the files from `/view`. Files land in `output/audio/moss/`.

## Tests

```
python -m unittest discover -s tests -v
```

They run the real bridge and worker over pipes with a fake model (`MOSS_SFX_FAKE=1`) — no GPU,
no MOSS. The node test needs any `torch` (CPU build is fine) and is skipped without it.

## Files

| File | Runs in | |
| --- | --- | --- |
| `__init__.py`, `nodes.py` | ComfyUI | the node |
| `bridge.py` | ComfyUI | config, worker process, protocol; standard library only |
| `worker.py` | MOSS venv | loads the model, generates; JSON lines on stdin/stdout, logs on stderr |

## Licence

The node: MIT. MOSS-SoundEffect v2 code and weights: Apache-2.0 (OpenMOSS).

---

## По-русски коротко

Узел ComfyUI для MOSS-SoundEffect v2. Модель работает в своём venv (у неё свои версии torch и
transformers), ComfyUI запускает её отдельным процессом. Установка — выше: venv с MOSS, клон этого
репозитория в `custom_nodes`, `config.json` с путём к `python.exe` из venv, перезапуск ComfyUI.
Узел — **audio → MOSS → MOSS Sound Effect**.
