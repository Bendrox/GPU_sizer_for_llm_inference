**What is vLLM.** A high-throughput LLM inference engine. Its key idea is **PagedAttention**:
the KV cache is split into small fixed-size **blocks of 16 tokens**, allocated on demand. This
cuts memory waste and lets the GPU serve many requests at once. This tab estimates **how many
concurrent requests** a GPU can hold for a given request length.

**Parameters**
- **GPU memory utilization** *(default 0.9)* — fraction of total VRAM vLLM may use; the rest is left for CUDA context and fragmentation. Higher = more room for KV cache, but riskier.
- **Sequence length per request** — tokens (prompt + generated) one request occupies; longer sequences cost more blocks, so fewer fit.
- **Model weight quantization** — precision of the weights (FP32=4, BF16=2, FP8=1 bytes). Lower → smaller model → more VRAM left for KV cache.
- **KV cache dtype** — precision of the KV cache, set **independently** from the weights (vLLM's `kv_cache_dtype`). FP8 KV roughly doubles capacity vs BF16; defaults to the model's precision (like `auto`).

**Quantization, in short.** Quantizing the **weights** frees VRAM by shrinking the model.
Quantizing the **KV cache** shrinks the per-token cost, so each request holds more tokens. The two are independent.

```text
usable_vram   = total_vram × gpu_memory_utilization
weights_bytes = params_billion × 1e9 × weight_bytes
kv_bytes      = usable_vram − weights_bytes        # if ≤ 0: does not fit

bytes_per_token = 2 × num_kv_heads × head_dim × num_layers × kv_dtype_bytes
block_bytes     = bytes_per_token × 16             # 16 tokens / block

num_blocks          = kv_bytes // block_bytes
total_kv_tokens     = num_blocks × 16
blocks_per_request  = ceil(seq_len / 16)
concurrent_requests = num_blocks // blocks_per_request
```
