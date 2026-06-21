---
title: GPU Sizer For LLM Inference
emoji: 🖥️
sdk: docker
app_port: 7860
pinned: false
---

# 🖥️🟩 GPU Memory Sizer API for LLM Inference

Estimate the **VRAM** an LLM needs for inference — KV cache, model weights, the longest context that fits on a given card, and a VRAM‑vs‑context chart — from the model's attention architecture alone.

The KV cache is derived from the model's architecture (KV heads, head dimension, number of attention layers) and reported for FP32, BF16 and FP8. You can also add the model weights to get the total memory actually used.

## Why this exists

Picking hardware for LLM serving means answering three recurring questions:

1. **How much VRAM** does my model need at a given context length and batch size?
2. **What's the longest context** I can serve on the GPU I already have?
3. **Where's the cliff** — at what context length does the KV cache overflow the card?

This tool answers all three from a model's attention shape (`num_kv_heads`, `head_dim`, `num_attention_layers`) and parameter count — no weights to download, no GPU required to run the estimate. 

Results are broken down across **FP32 / BF16 / FP8** so you can see the trade‑off of KV‑cache quantization at a glance.

## Features

| | Feature | Description |
|---|---|---|
| 🧠 | **KV cache calculator** | VRAM for a given context length, with optional model weights and batch size, across FP32/BF16/FP8. |
| 📏 | **Max context for a GPU** | The number of tokens that fit in the KV cache of a card with *N* GB of VRAM. |
| 📈 | **Context‑vs‑memory chart** | A PNG plotting total VRAM as context grows, one curve per precision, with NVIDIA card capacities (RTX 3090 → B200) drawn as reference lines. |

---


## Access 

**App access :** [Hugging Face Space](https://huggingface.co/spaces/eldiablo92/GPU_memory_sizer_for_LLM_inference)

## Model Catalog

A few models ship in `data/models.json` (Llama 3.x, Mistral 7B...). To add one, drop an entry with the same fields; they are loaded at startup.

## Notes

- Memory is in **MB base 1000** (the manufacturers' "on the box" convention), not MiB.
- Model quantization is set via `model_quantization_bytes`: `4` = FP32, `2` = BF16, `1` = FP8.
- The chart draws one curve per precision and places several NVIDIA GPU capacities (RTX 3090 → B200) as reference lines.


## CI dev

CI runs lint + tests on every push and PR to `main` (`.github/workflows/ci.yml`). 

