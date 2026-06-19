from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.schemas import ModelParams
from app.plotting import render_context_vs_memory

router = APIRouter(tags=["Plot"])


@router.post("/plot-context-vs-memory")
def plot_context_vs_memory(p: ModelParams, max_tokens: int = Query(default=131072, gt=0)):
    """PNG: total VRAM (weights + KV cache) vs number of tokens, per precision,
    with GPU capacities drawn as reference lines."""
    buf = render_context_vs_memory(p, max_tokens)
    return StreamingResponse(buf, media_type="image/png")
