**Parameters**
- **Context (tokens)** — sequence length; the KV cache grows linearly with it.
- **Batch size** — number of sequences in parallel; multiplies the KV cache.
- **Include model weights** — also add the model size (per precision) to get the total VRAM.
- **Add estimated CUDA overhead** — inflate the total by a fragmentation factor plus a fixed overhead.

```text
bytes_per_token = 2 × num_kv_heads × head_dim × num_layers
kv_cache_bytes  = bytes_per_token × context × batch × kv_dtype_bytes
weights_bytes   = params_billion × 1e9 × weight_bytes
total           = kv_cache_bytes + weights_bytes

# optional CUDA overhead
real = int((kv_mb + weights_mb) × (1 + frag)) + fixed_overhead_mb
```
