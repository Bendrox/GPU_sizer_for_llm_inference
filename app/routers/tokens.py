from fastapi import APIRouter
from schemas import ModeParams, TokenResult
from core import calcul_tokens

router = APIRouter(tags=["Tokens"])


@router.post("/max-context-len-4-GPU-memory", response_model=TokenResult)
def max_seq_len(p: ModeParams, vram_go: float) -> TokenResult:
    return calcul_tokens(p, vram_go)