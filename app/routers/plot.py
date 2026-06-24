from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.schemas import ModelParams
from app.plotting import render_context_vs_memory

router = APIRouter(tags=["Plot"])


@router.post("/plot-context-vs-memory")
def plot_context_vs_memory(
    p: ModelParams,
    max_tokens: int = Query(default=131072, gt=0),
    frag: float = Query(default=0.0, ge=0),
    fixed_overhead_mb: int = Query(default=0, ge=0),
):
    """PNG: total VRAM (weights + KV cache) vs number of tokens, per precision,
    with GPU capacities drawn as reference lines. Optionally adds estimated CUDA
    overhead: (weights + KV) * (1 + frag) + fixed_overhead_mb."""
    buf = render_context_vs_memory(p, max_tokens, frag, fixed_overhead_mb)
    return StreamingResponse(buf, media_type="image/png")
