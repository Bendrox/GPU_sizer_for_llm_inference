from pydantic import BaseModel, Field, field_validator
from app.config import UNIT_MEMORY, UNIT_TOKENS


class ModelParams(BaseModel):
    name: str
    publisher: str
    architecture: str
    num_kv_heads: int = Field(
        gt=0, description="Number of key/value (KV) attention heads"
    )
    head_dim: int = Field(gt=0, description="Dimension of each attention head")
    num_attention_layers: int = Field(
        gt=0,
        description="Number of attention layers. Note: exclude DeltaNet layers, which have no KV cache!",
    )
    total_params_billion: int
    model_quantization_bytes: int = Field(
        default=2, description="Bytes per stored model parameter: 4=fp32, 2=bf16, 1=fp8"
    )

    @field_validator("model_quantization_bytes")
    @classmethod
    def check_quant(cls, v):
        if v not in (1, 2, 4):
            raise ValueError("Quantization must be 1, 2, or 4")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Mistral-7B-v0.1",
                    "publisher": "Mistral AI",
                    "architecture": "Transformer",
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "num_attention_layers": 32,
                    "total_params_billion": 7,
                }
            ]
        }
    }


class KVCacheResult(BaseModel):
    memory_consumption_fp32_mb: int
    memory_consumption_bf16_mb: int
    memory_consumption_fp8_mb: int
    includes_model_weights: bool = False
    model_weights_mb: dict[str, int] = Field(
        default_factory=lambda: {"fp32": 0, "bf16": 0, "fp8": 0}
    )
    kv_cache_only_mb: dict[str, int]
    totals_mb: dict[str, dict[str, int]] = Field(default_factory=dict)
    unit: str = UNIT_MEMORY


class TokenResult(BaseModel):
    num_token_fp32_in_KVcache: str
    num_token_bf16_in_KVcache: str
    num_token_fp8_in_KVcache: str
    unit: str = UNIT_TOKENS
    vrm_enough_for_model:bool


class VLLMResult(BaseModel):
    fits: bool
    usable_vram_mb: int
    weights_mb: int
    kv_cache_mb: int
    block_size: int
    num_blocks: int
    total_tokens: int
    blocks_per_request: int
    max_concurrent_requests: int
    unit: str = UNIT_MEMORY