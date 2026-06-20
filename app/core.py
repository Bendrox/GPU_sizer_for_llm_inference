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
    """KV cache for `length_seq` tokens, in MB, for FP32/BF16/FP8.
    If include_model_weights, add the model weights (according to its quantization)."""
    elements = per_token_bytes_base(p) * length_seq * batch_size

    kv_only = {
        "fp32": bytes_to_mb(elements * 4),
        "bf16": bytes_to_mb(elements * 2),
        "fp8": bytes_to_mb(elements * 1),
    }

    model_bytes = p.total_params_billion * 1_000_000_000 * p.model_quantization_bytes
    model_weights = model_bytes if include_model_weights else 0

    return KVCacheResult(
        memory_consumption_fp32_mb=bytes_to_mb(elements * 4 + model_weights),
        memory_consumption_bf16_mb=bytes_to_mb(elements * 2 + model_weights),
        memory_consumption_fp8_mb=bytes_to_mb(elements * 1 + model_weights),
        includes_model_weights=include_model_weights,
        model_weights_mb=bytes_to_mb(model_bytes),
        kv_cache_only_mb=kv_only,
    )


def compute_max_tokens(p: ModelParams, vram_gb: float) -> TokenResult:
    """From VRAM (GB, base 1000) to the number of tokens storable in the KV cache."""
    per_token = per_token_bytes_base(p)
    vram_bytes = vram_gb * 1_000_000_000
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
    )
