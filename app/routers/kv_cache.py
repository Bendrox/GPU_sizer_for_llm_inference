from fastapi import APIRouter, Query
from app.schemas import ModelParams, KVCacheResult
from app.core import compute_kv_cache

router = APIRouter(tags=["KV Cache"])


@router.post("/kv-cache-size-calculator", response_model=KVCacheResult)
def kv_cache_size_calculator(
    params: ModelParams,
    include_model_weights: bool = False,
    length_seq: int = Query(gt=0),
    batch_size: int = Query(default=1, gt=0),
):
    return compute_kv_cache(params, length_seq, include_model_weights, batch_size)
