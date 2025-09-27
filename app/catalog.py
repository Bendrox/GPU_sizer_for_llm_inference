import json
from pathlib import Path
from app.schemas import ModeParams

_DATA_FILE = Path(__file__).parent / "data" / "models.json" # chemin relatif au fichier, pas au dossier de lancement

CATALOG = {m["nom"]: ModeParams(**m)
           for m in json.loads(_DATA_FILE.read_text(encoding="utf-8"))}