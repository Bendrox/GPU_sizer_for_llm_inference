import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans interface, obligatoire côté serveur
import matplotlib.pyplot as plt

from schemas import ModeParams
from core import per_token_bytes_base
from config import GPU_VRAM_GO


def render_context_vs_memory(p: ModeParams, max_tokens: int) -> io.BytesIO:
    per_token = per_token_bytes_base(p)
    model_go = p.total_params_billion * 1_000_000_000 * p.model_quantization_oct / 1e9

    tokens = np.linspace(0, max_tokens, 200)
    fig, ax = plt.subplots(figsize=(11, 6))

    for nom, oct_par_val in {"FP32": 4, "BF16": 2, "FP8": 1}.items():
        vram_go = model_go + (per_token * oct_par_val * tokens) / 1e9
        ax.plot(tokens, vram_go, linewidth=2, label=f"KV {nom} (+ poids modèle)")

    for nom, cap in sorted(GPU_VRAM_GO.items(), key=lambda x: x[1]):
        ax.axhline(cap, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(max_tokens, cap, f" {nom}", va="center", fontsize=8)

    ax.set_xlabel("Longueur de contexte (tokens)")
    ax.set_ylabel("VRAM totale requise (Go, base 1000)")
    ax.set_title(f"{p.nom} — VRAM vs contexte (poids modèle = {model_go:.1f} Go)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_tokens)
    ax.set_ylim(bottom=0)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # libère la mémoire, sinon fuite à chaque appel
    buf.seek(0)
    return buf