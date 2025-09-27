from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.schemas import ModeParams
from app.plotting import render_context_vs_memory

router = APIRouter(tags=["Plot"])


@router.post("/plot-context-vs-memory")
def plot_context_vs_memory(p: ModeParams, max_tokens: int = Query(default=131072, gt=0)):
    """PNG : VRAM totale (poids + KV cache) vs nombre de tokens, par précision,
    avec les capacités GPU en lignes de référence."""
    buf = render_context_vs_memory(p, max_tokens)
    return StreamingResponse(buf, media_type="image/png")