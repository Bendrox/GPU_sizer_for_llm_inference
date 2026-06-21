from fastapi import APIRouter
from app.schemas import ModelParams, TokenResult
from app.core import compute_max_tokens

router = APIRouter(tags=["Tokens"])


@router.post("/max-context-len-4-GPU-memory", response_model=TokenResult)
def max_seq_len(p: ModelParams, vram_gb: float, include_model_weights: bool) -> TokenResult:
    return compute_max_tokens(p, vram_gb, include_model_weights)
