from fastapi import APIRouter
from app.config import GPU_CATALOG

router = APIRouter(tags=["Catalog"])


@router.get("/gpus")
def get_gpus() -> dict[str, float]:
    """GPU catalog: name -> VRAM in GB (base 1000)."""
    return GPU_CATALOG
