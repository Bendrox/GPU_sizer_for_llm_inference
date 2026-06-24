import io
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend, required on the server side
import matplotlib.pyplot as plt

from app.schemas import ModelParams
from app.core import per_token_bytes_base
from app.config import GPU_VRAM_GB


def render_context_vs_memory(
    p: ModelParams, max_tokens: int, frag: float = 0.0, fixed_overhead_mb: int = 0
) -> io.BytesIO:
    per_token = per_token_bytes_base(p)
    model_gb = p.total_params_billion * 1_000_000_000 * p.model_quantization_bytes / 1e9

    tokens = np.linspace(0, max_tokens, 200)
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, bytes_per_val in {"FP32": 4, "BF16": 2, "FP8": 1}.items():
        vram_gb = model_gb + (per_token * bytes_per_val * tokens) / 1e9
        # estimated real allocation: (weights + KV) * (1 + frag) + fixed CUDA overhead
        vram_gb = vram_gb * (1 + frag) + fixed_overhead_mb / 1000
        ax.plot(tokens, vram_gb, linewidth=2, label=f"KV {name} (+ model weights)")

    for name, cap in sorted(GPU_VRAM_GB.items(), key=lambda x: x[1]):
        ax.axhline(cap, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(max_tokens, cap, f" {name}", va="center", fontsize=8)

    ax.set_xlabel("Length of context (tokens)")
    ax.set_ylabel("Total VRAM required (GB, base 1000)")
    ax.set_title(f"{p.name} — VRAM vs context (model weights = {model_gb:.1f} GB)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_tokens)
    ax.set_ylim(bottom=0)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # free the memory, otherwise a leak on every call
    buf.seek(0)
    return buf
