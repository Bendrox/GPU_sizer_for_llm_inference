from schemas import ModeParams, KVCacheResult, TokenResult
from units import oct_to_mo, format_large_numbers


def per_token_bytes_base(p: ModeParams) -> int:
    """Octets par token, toutes couches, hors quantification (K + V)."""
    return 2 * p.nbr_attention_heads_kv * p.nbr_head_dim * p.nbr_Gated_Attention_layers


def calculer_kv_cache(p: ModeParams, length_seq: int, include_model_weights: bool = False) -> KVCacheResult:
    """KV cache pour `length_seq` tokens, en Mo, pour FP32/BF16/FP8.
    Si include_model_weights, ajoute le poids modèle (selon sa quantification)."""
    elements = per_token_bytes_base(p) * length_seq

    kv_only = {"fp32": oct_to_mo(elements * 4),
               "bf16": oct_to_mo(elements * 2),
               "fp8":  oct_to_mo(elements * 1)}

    model_bytes = p.total_params_billion * 1_000_000_000 * p.model_quantization_oct
    extra = model_bytes if include_model_weights else 0

    return KVCacheResult(
        memory_consumption_fp32_mo=oct_to_mo(elements * 4 + extra),
        memory_consumption_bf16_mo=oct_to_mo(elements * 2 + extra),
        memory_consumption_fp8_mo=oct_to_mo(elements * 1 + extra),
        includes_model_weights=include_model_weights,
        model_weights_mo=oct_to_mo(model_bytes),
        kv_cache_only_mo=kv_only,
    )


def calcul_tokens(p: ModeParams, vram_go: float) -> TokenResult:
    """De la VRAM (Go, base 1000) vers le nombre de tokens stockables en KV cache."""
    per_token = per_token_bytes_base(p)
    vram_bytes = vram_go * 1_000_000_000
    return TokenResult(
        num_token_fp32_in_KVcache=format_large_numbers(int(vram_bytes // (per_token * 4))),
        num_token_bf16_in_KVcache=format_large_numbers(int(vram_bytes // (per_token * 2))),
        num_token_fp8_in_KVcache=format_large_numbers(int(vram_bytes // (per_token * 1))),
    )