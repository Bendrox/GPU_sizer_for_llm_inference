GPU_VRAM_GB = {  # VRAM in GB (base 1000, "on the box")
    "RTX 3090 (24)": 24,
    "RTX 4090 (24)": 24,
    "RTX 5090 (32)": 32,
    "A100 (80)": 80,
    "H100 (80)": 80,
    "H200 (141)": 141,
    "B200 (192)": 192,
}

GPU_CATALOG = {  # served to the UI dropdown via GET /gpus -> VRAM in GB (base 1000)
    "NVIDIA H200 (141 GB)": 141.0,
    "NVIDIA H100 (80 GB)": 80.0,
    "NVIDIA A100 (40 GB)": 40.0,
    "NVIDIA RTX A6000 (48 GB)": 48.0,
    "NVIDIA V100 (32 GB)": 32.0,
    "NVIDIA L4 (24 GB)": 24.0,
    "NVIDIA RTX 4090 (24 GB)": 24.0,
    "NVIDIA RTX 3090 (24 GB)": 24.0,
}

UNIT_MEMORY = "MB (base 1000, decimal)"
UNIT_TOKENS = "tokens"
