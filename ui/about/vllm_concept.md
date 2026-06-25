**vLLM** is a high-throughput LLM inference engine. Its key idea is **PagedAttention**: the KV
cache is split into small fixed-size **blocks of 16 tokens**, allocated on demand instead of one
big contiguous buffer per request. This cuts memory waste and lets the GPU serve many requests at
once.

This tab estimates **how many concurrent requests** a GPU can hold for a given request length, and
tells you clearly when the model weights do not even fit.
