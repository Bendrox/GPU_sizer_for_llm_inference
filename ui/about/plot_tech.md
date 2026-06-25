**Parameters**
- **Max context on X axis** — upper bound of the context length swept on the horizontal axis.
- **Add estimated CUDA overhead** — lift every curve by a fragmentation factor plus a fixed overhead.

```text
weights_gb      = params_billion × weight_bytes          # base 1000
bytes_per_token = 2 × num_kv_heads × head_dim × num_layers
vram_gb(tokens) = weights_gb + bytes_per_token × tokens × kv_dtype_bytes / 1e9

# optional CUDA overhead
vram_gb = vram_gb × (1 + frag) + fixed_overhead_mb / 1000
```
