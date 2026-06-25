**What it does.** The **inverse** of the KV cache tab: given a GPU's VRAM, it computes **how many
tokens fit** in the KV cache once the model weights are loaded, for each KV precision
(FP32/BF16/FP8). The donuts show the weights vs KV-cache share of VRAM, with the token count in
the center.

**Parameters**
- **VRAM source** — type an amount in GB, or pick a GPU from the list.
- **Model weight quantization** — precision of the weights; larger weights leave less room for the KV cache.
- **Include model weights** — subtract the model size from VRAM before counting tokens.
- **Add estimated CUDA overhead** — reserve VRAM (fragmentation + fixed) before counting.

```text
vram_bytes      = vram_gb × 1e9
weights_bytes   = params_billion × 1e9 × weight_bytes   # if included
available       = vram_bytes − weights_bytes            # if < 0: does not fit
bytes_per_token = 2 × num_kv_heads × head_dim × num_layers
max_tokens      = available // (bytes_per_token × kv_dtype_bytes)
```
