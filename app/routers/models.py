from fastapi import APIRouter, HTTPException
from app.schemas import ModeParams
from app.catalog import CATALOG

router = APIRouter(tags=["Catalog"])


@router.get("/models")
def get_all_models() -> list[str]:
    """Liste available models"""
    return list(CATALOG.keys())


@router.get("/models/{nom}", response_model=ModeParams)
def get_model_config(nom: str):
    """Compelte model config (for others endpoints)."""
    if nom not in CATALOG:
        raise HTTPException(status_code=404, detail=f"Modèle '{nom}' introuvable")
    return CATALOG[nom]