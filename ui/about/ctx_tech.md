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
