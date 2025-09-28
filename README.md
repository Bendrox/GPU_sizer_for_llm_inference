# 🖥️🟩 GPU Memory Sizer API for LLM Inference

A small API to estimate the VRAM needed for LLM inference: KV cache size, the maximum context that fits on a given card, and a VRAM-vs-context chart.

The KV cache is derived from the model's architecture (KV heads, head dimension, number of attention layers) and reported for FP32, BF16 and FP8. You can also add the model weights to get the total memory actually used.

## 🔌 Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/health` | Ping the API |
| `GET`  | `/models` | List the catalog models |
| `GET`  | `/models/{nom}` | Full config of a model |
| `POST` | `/kv-cache-size-calculator` | Memory for a given context (optional: + model weights, batch) |
| `POST` | `/max-context-len-4-GPU-memory` | Number of tokens that fit in a given VRAM |
| `POST` | `/plot-context-vs-memory` | PNG chart: total VRAM as a function of context |

Interactive docs are available at `/docs` once the API is running.

## 🏃 Running

```bash
uv add fastapi uvicorn pydantic numpy matplotlib
uvicorn app.app:app --reload
```

The folder holding the code must be importable as `app` (imports use `from app...`).

## 📑 Catalog

A few models ship in `data/models.json` (Llama 3.x, Mistral 7B...). To add one, drop an entry with the same fields; they are loaded at startup.

## 📑 Notes

- Memory is in **MB base 1000** (the manufacturers' "on the box" convention), not MiB.
- Model quantization is set via `model_quantization_oct`: `4` = FP32, `2` = BF16, `1` = FP8.
- The chart draws one curve per precision and places several NVIDIA GPU capacities (RTX 3090 → B200) as reference lines.

## Structure

```
app.py        FastAPI entry point
core.py       KV cache / token calculations
plotting.py   chart generation
catalog.py    model loading
config.py     GPU capacities and units
schemas.py    Pydantic models
units.py      bytes / MB conversions
routers/      one file per group of endpoints
data/         model catalog (JSON)
```
## Frontend 
