from fastapi import FastAPI
from app.routers import kv_cache, tokens, plot, health, models  

app = FastAPI(
    title="GPU Memory Sizer API for LLM Inference",
    description="""
### API to calculate memory needed for KV during LLM inference.

#### KV cache section :
- KV Cache Calculator : Calculate the KV Cache size based on the length of your contexts (tokens), model's architecture, and precision for both KV cache and the model FP32, BF16, FP8 precision
#### Token section : 
- Max Sequence Length : Calculates the maximum context size supported by a given hardware configuration (based on GPU memory) including your language model weight.
#### Plot section : 
- Context vs Memory Plot : Returns a PNG chart showing total VRAM (model weights + KV cache) as a function of context length, with one curve per precision (FP32, BF16, FP8) and the memory capacity of several NVIDIA GPUs (consumer and data center) drawn as reference lines.
#### Catalog section : 
- Get all models that are already configured in the app.
- Get a model parameters based on the name.
""",
)

app.include_router(kv_cache.router)
app.include_router(tokens.router)
app.include_router(plot.router)
app.include_router(models.router)
app.include_router(health.router)