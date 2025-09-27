from fastapi import APIRouter, Query
from app.schemas import ModeParams, KVCacheResult
from app.core import calculer_kv_cache

router = APIRouter(tags=["KV Cache"])


@router.post("/kv-cache-size-calculator", response_model=KVCacheResult)
def kv_cache_size_calculator(params: ModeParams, include_model_weights: bool = False, length_seq: int = Query(gt=0)):
    return calculer_kv_cache(params, length_seq, include_model_weights)