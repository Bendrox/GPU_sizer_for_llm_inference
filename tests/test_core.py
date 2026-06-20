from app.schemas import ModelParams
from app.core import per_token_bytes_base
from app.core import compute_kv_cache
from app.core import compute_max_tokens

import pytest
from pydantic import ValidationError

mistral = ModelParams(
    name="Mistral-7B-v0.1",
    publisher="Mistral AI",
    architecture="Transformer (GQA)",
    num_kv_heads=8,
    head_dim=128,
    num_attention_layers=32,
    total_params_billion=7,
    model_quantization_bytes=2,
)


def test_obvious():
    assert 1 + 1 == 2


def test_per_token_bytes_base():
    assert per_token_bytes_base(mistral) == 65536


def test_kv_cache_only():
    res = compute_kv_cache(mistral, length_seq=8192)
    assert res.kv_cache_only_mb == {"fp32": 2147, "bf16": 1073, "fp8": 536}
    assert res.includes_model_weights is False


def test_kv_cache_with_weights():
    res = compute_kv_cache(mistral, length_seq=8192, include_model_weights=True)
    assert res.includes_model_weights is True
    assert res.memory_consumption_bf16_mb == 15073  # 1073 KV + 14000 poids


def test_compute_max_tokens():
    res = compute_max_tokens(mistral, vram_gb=80)
    assert res.num_token_bf16_in_KVcache == "610 351"


def test_invalid_quantization():
    with pytest.raises(ValidationError):
        ModelParams(
            name="x",
            publisher="y",
            architecture="z",
            num_kv_heads=8,
            head_dim=128,
            num_attention_layers=32,
            total_params_billion=7,
            model_quantization_bytes=3,
        )
