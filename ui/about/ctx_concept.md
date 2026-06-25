This is the **inverse** of the KV cache tab. Given a GPU's VRAM, it answers: once the model
weights are loaded, **how many tokens fit** in the KV cache? The answer is reported for each KV
precision (FP32 / BF16 / FP8).

The donuts show how the VRAM splits between **model weights** and **KV cache**, with the maximum
token count shown in the center.
