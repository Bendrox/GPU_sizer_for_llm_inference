import json
from pathlib import Path
from app.schemas import ModelParams

_DATA_FILE = Path(__file__).parent / "data" / "models.json"  # path relative to this file, not the launch directory

CATALOG = {m["name"]: ModelParams(**m)
           for m in json.loads(_DATA_FILE.read_text(encoding="utf-8"))}
