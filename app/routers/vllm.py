from fastapi import APIRouter, Query
from app.schemas import ModelParams, VLLMResult
from app.core import compute_vllm_capacity

router = APIRouter(tags=["vLLM"])


@router.post("/vllm-capacity", response_model=VLLMResult)
def vllm_capacity(
    params: ModelParams,
    total_vram_gb: float = Query(gt=0),
    seq_len: int = Query(gt=0),
    gpu_memory_utilization: float = Query(default=0.9, gt=0, le=1),
    block_size: int = Query(default=16, gt=0),
    kv_dtype_bytes: int = Query(default=2, gt=0),
):
    return compute_vllm_capacity(
        params, total_vram_gb, seq_len, gpu_memory_utilization, block_size, kv_dtype_bytes
    )
