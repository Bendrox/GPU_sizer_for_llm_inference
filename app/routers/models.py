from fastapi import APIRouter, HTTPException
from app.schemas import ModelParams
from app.catalog import CATALOG

router = APIRouter(tags=["Catalog"])


@router.get("/models")
def get_all_models() -> list[str]:
    """List available models."""
    return list(CATALOG.keys())


@router.get("/models/{name}", response_model=ModelParams)
def get_model_config(name: str):
    """Complete model config (used by the other endpoints)."""
    if name not in CATALOG:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return CATALOG[name]
