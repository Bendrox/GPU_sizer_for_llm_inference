from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator

app=FastAPI()


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

class KVCacheResult(BaseModel):
    resultat:int


def calculer_kv_cache(p:ModeParams, lenght_seq: int ) -> KVCacheResult :
        # step 1 : calculer éléments par token et par couche
        elmt_per_token_per_layer=2*p.nbr_attention_heads_kv*p.nbr_head_dim
        # step 2: multiplié par tokens et couches
        # Chaque bloc contient 1 couche d'attention, le bloc est répété N fois (nbr_Gated_Attention_layers)
        elements=elmt_per_token_per_layer*lenght_seq*(p.nbr_Gated_Attention_layers)
        # step 3 : Des éléments aux octets
        kv_cache_oct=elements*p.quantization_oct
        output=kv_cache_oct/(2**20)

        return KVCacheResult(resultat=output)

@app.post("/kv-cache-calculator", response_model=KVCacheResult)
def endpoint_kv_cache_calculator(params:ModeParams, lenght_seq :int =Query(gt=0)):
      return calculer_kv_cache(params, lenght_seq)
