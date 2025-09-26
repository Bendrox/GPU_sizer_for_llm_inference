from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator


import io
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend sans interface, obligatoire côté serveur
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse


app=FastAPI(title="TensorSizer - Inference Memory API",
            description="""
    API to calculate memory needed for KV during LLM inference.
    
    Features :
    - KV Cache Calculator : Calculate the KV Cache size based on the lenght of your contexts (tokents), model's architecture , and precision for both KV cache and the model FP32, BF16 , FP8 precision
    - Max Sequence Length : Calculates the maximum context size supported by a given hardware configuration (based on GPU memory) including your language model weight.
    - Context vs Memory Plot : Returns a PNG chart showing total VRAM (model weights + KV cache) as a function of context length, with one curve per precision (FP32, BF16, FP8) and the memory capacity of several NVIDIA GPUs (consumer and data center) drawn as reference lines.
    """,)


GPU_VRAM_GO = { # VRAM en Go (base 1000, "sur la boîte")
    "RTX 3090 (24)": 24,
    "RTX 4090 (24)": 24,
    "RTX 5090 (32)": 32,
    "A100 (80)": 80,
    "H100 (80)": 80,
    "H200 (141)": 141,
    "B200 (192)": 192,
}



class ModeParams(BaseModel):
        nom:str
        editeur:str
        archi_type:str
        nbr_attention_heads_kv: int =Field(gt=0, description="Number of Attention Heads for KV")
        nbr_head_dim: int =Field(gt=0)
        nbr_Gated_Attention_layers: int = Field(gt=0, description="nombre de couches Gated Attention, déduit du Hidden Layout. Attention pas de DeltaNet qui n ont pas de KV cache!")
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
                        "total_params_billion":7
                    }
                ]
            }
        }

class KVCacheResult(BaseModel):
    memory_consumption_fp32_mo:int
    memory_consumption_bf16_mo:int
    memory_consumption_fp8_mo:int
    includes_model_weights: bool = False
    model_weights_mo: int = 0
    kv_cache_only_mo: dict
    unit: str = "Mo (base 1000, décimale)"

class TokenResult(BaseModel):
    num_token_fp32_in_KVcache:str
    num_token_bf16_in_KVcache:str
    num_token_fp8_in_KVcache:str
    unit: str = "tokens"

def format_large_numbers(large_num):
      return f"{large_num:,}".replace(",", " ")

def oct_to_mo(input: int):
      """Octets -> Mo (base 1000). Convention 'sur la boîte' utilisée par les fabriquants """
      return input// (10**6)

def oct_to_mio(input: int) -> int:
    """Octets -> Mio en base 1024 (binaire). Convention 'à l'affichage' utilisée par les outils GPU"""
    return input // (2**20)

def calculer_kv_cache(p:ModeParams, lenght_seq: int , include_model_weights: bool = False) -> KVCacheResult :
        """Calcule la mémoire du KV cache pour une séquence de `length_seq` tokens, exprimée en Mo pour les trois précisions FP32, BF16 et FP8
        """
        # step 1 : calculer éléments par token et par couche
        elmt_per_token_per_layer=2*p.nbr_attention_heads_kv*p.nbr_head_dim
        
        # step 2: multiplié par tokens et couches, le bloc est répété N fois (nbr_Gated_Attention_layers)
        elements=elmt_per_token_per_layer*lenght_seq*(p.nbr_Gated_Attention_layers)
        
        # KV cache seul, indépendant du flag
        kv_only_fp32 = oct_to_mo(elements * 4)
        kv_only_bf16 = oct_to_mo(elements * 2)
        kv_only_fp8  = oct_to_mo(elements * 1)

        model_bytes = p.total_params_billion * 1_000_000_000 * p.model_quantization_oct

        kv_bytes_fp32 = elements * 4
        kv_bytes_bf16 = elements * 2
        kv_bytes_fp8  = elements * 1
        
        if include_model_weights:
            kv_bytes_fp32 += model_bytes
            kv_bytes_bf16 += model_bytes
            kv_bytes_fp8  += model_bytes

        mo_fp32 = oct_to_mo(kv_bytes_fp32)
        mo_bf16 = oct_to_mo(kv_bytes_bf16)
        mo_fp8  = oct_to_mo(kv_bytes_fp8)

        return KVCacheResult(
            memory_consumption_fp32_mo=mo_fp32,
            memory_consumption_bf16_mo=mo_bf16,
            memory_consumption_fp8_mo=mo_fp8,
            includes_model_weights=include_model_weights,
            model_weights_mo=oct_to_mo(model_bytes),
            kv_cache_only_mo={"fp32": kv_only_fp32, "bf16": kv_only_bf16, "fp8": kv_only_fp8},
        )


def calcul_tokens(p:ModeParams, vram_go: float ) -> TokenResult :
        """ Calculate from GPU memory and your model weights to number of tokens possible to put in KV cache
        """
        # Calcul de la base par token (sans quantification)
        per_token = 2* p.nbr_attention_heads_kv * p.nbr_head_dim * p.nbr_Gated_Attention_layers 

        # Calcul du poids par token pour chaque précision (en octets)
        per_token_fp32 = per_token * 4
        per_token_bf16 = per_token * 2
        per_token_fp8  = per_token * 1     

        # Gio -> octets
        vram_bytes = vram_go * 1_000_000_000  

        return TokenResult(
              num_token_fp32_in_KVcache=format_large_numbers(int( vram_bytes // per_token_fp32)),
              num_token_bf16_in_KVcache=format_large_numbers(int( vram_bytes // per_token_bf16)),
              num_token_fp8_in_KVcache=format_large_numbers(int( vram_bytes // per_token_fp8)),
        )


@app.post("/kv-cache-size-calculator", response_model=KVCacheResult)
def kv_cache_size_calculator(params:ModeParams, include_model_weights : bool = False, lenght_seq :int =Query(gt=0) ):
      return calculer_kv_cache(params, lenght_seq, include_model_weights)

@app.post("/max-context-len-4-GPU-memory", response_model=TokenResult)      
def max_seq_len(p: ModeParams, vram_go: float) -> TokenResult:
        return calcul_tokens(p, vram_go)


@app.post("/plot-context-vs-memory")
def plot_context_vs_memory(p: ModeParams, max_tokens: int = Query(default=131072, gt=0)):
    """Renvoie un PNG : VRAM totale (poids + KV cache) en fonction du nombre
    de tokens, une courbe par précision, avec les capacités GPU en lignes."""

    # octets par token et par précision (K + V, toutes couches)
    per_token = 2 * p.nbr_attention_heads_kv * p.nbr_head_dim * p.nbr_Gated_Attention_layers
    model_go = p.total_params_billion * 1_000_000_000 * p.model_quantization_oct / 1e9

    tokens = np.linspace(0, max_tokens, 200)
    precisions = {"FP32": 4, "BF16": 2, "FP8": 1}

    fig, ax = plt.subplots(figsize=(11, 6))

    for nom, oct_par_val in precisions.items():
        vram_go = model_go + (per_token * oct_par_val * tokens) / 1e9
        ax.plot(tokens, vram_go, linewidth=2, label=f"KV {nom} (+ poids modèle)")

    for nom, cap in sorted(GPU_VRAM_GO.items(), key=lambda x: x[1]):
        ax.axhline(cap, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(max_tokens, cap, f" {nom}", va="center", fontsize=8)

    ax.set_xlabel("Longueur de contexte (tokens)")
    ax.set_ylabel("VRAM totale requise (Go, base 1000)")
    ax.set_title(f"{p.nom} — VRAM vs contexte (poids modèle = {model_go:.1f} Go)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_tokens)
    ax.set_ylim(bottom=0)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)               # libère la mémoire, sinon fuite à chaque appel
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



@app.get("/health")
def health():
    return {"status": "ok"}

