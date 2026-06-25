This tab plots **total VRAM (model weights + KV cache) as a function of context length**, with one
curve per KV precision (FP32 / BF16 / FP8). Horizontal lines mark the VRAM of common GPUs, so you
can read off the longest context each GPU can hold before running out of memory.
