from app.schemas import ModelParams, KVCacheResult, TokenResult
from app.units import bytes_to_mb, format_large_numbers


def per_token_bytes_base(p: ModelParams) -> int:
    """Bytes per token, across all layers, excluding quantization (K + V)."""
    return 2 * p.num_kv_heads * p.head_dim * p.num_attention_layers


def compute_kv_cache(
    p: ModelParams,
    length_seq: int,
    include_model_weights: bool = False,
    batch_size: int = 1,
) -> KVCacheResult:
    """KV cache for `length_seq` tokens, in MB. Returns a matrix `totals_mb`:
    one row per model-weight quantization (FP32/BF16/FP8), and for each row the 3
    KV-cache levels according to their own quantization."""
    elements = per_token_bytes_base(p) * length_seq * batch_size
    params = p.total_params_billion * 1_000_000_000
    quant_bytes = {"fp32": 4, "bf16": 2, "fp8": 1}

    kv_only = {q: bytes_to_mb(elements * b) for q, b in quant_bytes.items()}

    weights_bytes = (
        {q: params * b for q, b in quant_bytes.items()}
        if include_model_weights
        else {q: 0 for q in quant_bytes}
    )

    # rows = model-weight quantization, columns = KV-cache quantization
    totals_mb = {
        wq: {
            kvq: bytes_to_mb(elements * kvb + weights_bytes[wq])
            for kvq, kvb in quant_bytes.items()
        }
        for wq in quant_bytes
    }

    return KVCacheResult(
        memory_consumption_fp32_mb=totals_mb["fp32"]["fp32"],
        memory_consumption_bf16_mb=totals_mb["bf16"]["bf16"],
        memory_consumption_fp8_mb=totals_mb["fp8"]["fp8"],
        includes_model_weights=include_model_weights,
        model_weights_mb={k: bytes_to_mb(v) for k, v in weights_bytes.items()},
        kv_cache_only_mb=kv_only,
        totals_mb=totals_mb,
    )


def compute_max_tokens(p: ModelParams, vram_gb: float, include_model_weights: bool = False) -> TokenResult:
    """From VRAM (GB, base 1000) to the number of tokens storable in the KV cache."""
    per_token = per_token_bytes_base(p)
    vram_bytes = vram_gb * 1_000_000_000
    
    model_bytes = p.total_params_billion * 1_000_000_000 * p.model_quantization_bytes
    model_weights = model_bytes if include_model_weights else 0
    
    vram_bytes = vram_bytes- model_weights
    if vram_bytes <0:
        return TokenResult(num_token_fp32_in_KVcache="0",
                           num_token_bf16_in_KVcache="0",
                           num_token_fp8_in_KVcache="0",
                           vrm_enough_for_model=False)
    else: 
        return TokenResult(
            num_token_fp32_in_KVcache=format_large_numbers(
                int(vram_bytes // (per_token * 4))
            ),
            num_token_bf16_in_KVcache=format_large_numbers(
                int(vram_bytes // (per_token * 2))
            ),
            num_token_fp8_in_KVcache=format_large_numbers(
                int(vram_bytes // (per_token * 1))
            ),
            vrm_enough_for_model=True
        )
