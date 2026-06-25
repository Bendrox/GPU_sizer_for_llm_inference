The **KV cache** stores the Keys and Values of every past token during generation, so the model
does not recompute them at each step. It is the main *variable* memory cost at inference: it grows
**linearly** with the context length and the batch size.

This tab estimates that memory for the **same model declined in 3 precisions** (FP32 / BF16 / FP8),
and can add the model weights to show the **total VRAM** needed.
