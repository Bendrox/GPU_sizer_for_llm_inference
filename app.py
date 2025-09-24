from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator

app=FastAPI(title="Inference ")


class ModeParams(BaseModel):
        nom:str
        editeur:str
        archi_type:str
        nbr_attention_heads_kv: int =Field(gt=0, description="Number of Attention Heads for KV")
        nbr_head_dim: int =Field(gt=0)
        nbr_Gated_Attention_layers: int = Field(gt=0, description="nombre de couches Gated Attention, déduit du Hidden Layout. Attention pas de DeltaNet qui n ont pas de KV cache!")
        quantization_oct: int = Field(description="4=fp32, 2=bf16, 1=fp8") #Literal[1, 2, 4]

        @field_validator("quantization_oct")
        @classmethod
        def check_quant(cls, v):
              if v not in (1,2,4):
                    raise ValueError("Quantization doit etre 1,2 ou 4")
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
                        "quantization_oct": 2
                    }
                ]
            }
        }

class KVCacheResult(BaseModel):
    memory_consumption_fp32:int
    memory_consumption_bf16:int
    memory_consumption_fp8:int


def calculer_kv_cache(p:ModeParams, lenght_seq: int ) -> KVCacheResult :
        # step 1 : calculer éléments par token et par couche
        elmt_per_token_per_layer=2*p.nbr_attention_heads_kv*p.nbr_head_dim
        # step 2: multiplié par tokens et couches
        # Chaque bloc contient 1 couche d'attention, le bloc est répété N fois (nbr_Gated_Attention_layers)
        elements=elmt_per_token_per_layer*lenght_seq*(p.nbr_Gated_Attention_layers)
        # step 3 : calcul des octets pour chaque précision 
        bytes_fp32 = elements * 4
        bytes_bf16 = elements * 2
        bytes_fp8  = elements * 1
        # Étape 4 : Convertir en Mo
        mo_fp32 = bytes_fp32 // (2**20)
        mo_bf16 = bytes_bf16 // (2**20)
        mo_fp8  = bytes_fp8 // (2**20)

        return KVCacheResult(memory_consumption_fp32=mo_fp32,
                             memory_consumption_bf16=mo_bf16, 
                             memory_consumption_fp8=mo_fp8 )

@app.post("/kv-cache-calculator", response_model=KVCacheResult)
def endpoint_kv_cache_calculator(params:ModeParams, lenght_seq :int =Query(gt=0)):
      return calculer_kv_cache(params, lenght_seq)

@app.post("/max-len-4-GPU-memory", response_model=KVCacheResult)      
def endpoint_max_seq_len(p: ModeParams, vram_gib: float) -> int:
        per_token = 2 * p.nbr_attention_heads_kv * p.nbr_head_dim * p.nbr_Gated_Attention_layers * p.quantization_oct
        return int((vram_gib * 2**30) // per_token)
