from pydantic import BaseModel, Field, field_validator
from app.config import UNIT_MEMORY, UNIT_TOKENS


class ModeParams(BaseModel):
    nom: str
    editeur: str
    archi_type: str
    nbr_attention_heads_kv: int = Field(gt=0, description="Number of Attention Heads for KV")
    nbr_head_dim: int = Field(gt=0)
    nbr_Gated_Attention_layers: int = Field(gt=0, description="nombre de couches Gated Attention. Attention pas de DeltaNet qui n'ont pas de KV cache!")
    total_params_billion: int
    model_quantization_oct: int = Field(default=2, description="Octets par paramètre du modèle stocké: 4=fp32, 2=bf16, 1=fp8")

    @field_validator("model_quantization_oct")
    @classmethod
    def check_quant(cls, v):
        if v not in (1, 2, 4):
            raise ValueError("Quantization doit etre 1, 2 ou 4")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "nom": "Mistral-7B-v0.1",
                    "editeur": "Mistral AI",
                    "archi_type": "Transformer",
                    "nbr_attention_heads_kv": 8,
                    "nbr_head_dim": 128,
                    "nbr_Gated_Attention_layers": 32,
                    "total_params_billion": 7,
                }
            ]
        }
    }


class KVCacheResult(BaseModel):
    memory_consumption_fp32_mo: int
    memory_consumption_bf16_mo: int
    memory_consumption_fp8_mo: int
    includes_model_weights: bool = False
    model_weights_mo: int = 0
    kv_cache_only_mo: dict[str, int]
    unit: str = UNIT_MEMORY


class TokenResult(BaseModel):
    num_token_fp32_in_KVcache: str
    num_token_bf16_in_KVcache: str
    num_token_fp8_in_KVcache: str
    unit: str = UNIT_TOKENS