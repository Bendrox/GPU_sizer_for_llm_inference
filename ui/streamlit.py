import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="GPU Sizer", page_icon="🧮", layout="wide")
st.title("🧮 GPU Sizer for LLM inference")


#  Sidebar: API config + /health
with st.sidebar:
    st.header("⚙️ API")
    API = st.text_input("API URL", API)

    if st.button("Ping /health", use_container_width=True):
        try:
            r = requests.get(f"{API}/health", timeout=5)
            st.success(r.json()) if r.ok else st.error(r.text)
        except requests.RequestException as e:
            st.error(f"Unreachable: {e}")


#  Loading the catalog: GET /models
try:
    models = requests.get(f"{API}/models", timeout=5).json()
except requests.RequestException as e:
    st.error(f"Unable to load models from {API}: {e}")
    st.stop()


#  Model selection + editable config
st.subheader("📦 Model")
col_sel, col_quant = st.columns([2, 1])

name = col_sel.selectbox("Catalog", models)
cfg = requests.get(f"{API}/models/{name}").json()

with st.expander("Model architecture (editable)", expanded=False):
    c1, c2, c3 = st.columns(3)
    cfg["num_kv_heads"] = c1.number_input(
        "KV attention heads", min_value=1, value=cfg["num_kv_heads"]
    )
    cfg["head_dim"] = c2.number_input(
        "Dimension per head", min_value=1, value=cfg["head_dim"]
    )
    cfg["num_attention_layers"] = c3.number_input(
        "Attention layers", min_value=1, value=cfg["num_attention_layers"]
    )

    c4, c5 = st.columns(2)
    cfg["total_params_billion"] = c4.number_input(
        "Parameters (billions)", min_value=1, value=cfg["total_params_billion"]
    )
    quant_label = {4: "FP32 (4)", 2: "BF16 (2)", 1: "FP8 (1)"}
    cfg["model_quantization_bytes"] = c5.selectbox(
        "Weight quantization",
        [4, 2, 1],
        index=[4, 2, 1].index(cfg.get("model_quantization_bytes", 2)),
        format_func=lambda v: quant_label[v],
    )

    st.json(cfg)


#  Tabs: one per POST endpoint
tab_kv, tab_ctx, tab_plot = st.tabs(
    ["🧠 KV cache", "📏 Max context / GPU", "📈 Context vs memory plot"]
)


#  POST /kv-cache-size-calculator
with tab_kv:
    st.caption("Memory for a given context (option: + model weights, batch).")
    c1, c2, c3 = st.columns(3)
    ctx = c1.number_input("Context (tokens)", min_value=1, value=8192, step=1024)
    batch = c2.number_input("Batch size", min_value=1, value=1)
    inc_weights = c3.checkbox("Include model weights", value=True)

    if st.button("Compute KV cache", type="primary", key="btn_kv"):
        r = requests.post(
            f"{API}/kv-cache-size-calculator",
            params={
                "length_seq": ctx,
                "batch_size": batch,
                "include_model_weights": inc_weights,
            },
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            kv = r.json()
            label = "total (weights + KV)" if inc_weights else "KV cache only"
            st.markdown(f"**{label} memory**")
            m1, m2, m3 = st.columns(3)
            m1.metric("FP32", f"{kv['memory_consumption_fp32_mb']:,} MB")
            m2.metric("BF16", f"{kv['memory_consumption_bf16_mb']:,} MB")
            m3.metric("FP8", f"{kv['memory_consumption_fp8_mb']:,} MB")

            if kv["includes_model_weights"]:
                st.caption(f"Model weights: {kv['model_weights_mb']:,} MB")
            with st.expander("Raw response"):
                st.json(kv)


#  POST /max-context-len-4-GPU-memory
with tab_ctx:
    st.caption("Number of tokens storable in KV cache for a given VRAM.")
    vram = st.number_input("Available VRAM (GB)", min_value=1.0, value=24.0, step=1.0)

    if st.button("Compute max context", type="primary", key="btn_ctx"):
        r = requests.post(
            f"{API}/max-context-len-4-GPU-memory",
            params={"vram_gb": vram},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            tok = r.json()
            m1, m2, m3 = st.columns(3)
            m1.metric("FP32", f"{tok['num_token_fp32_in_KVcache']} tokens")
            m2.metric("BF16", f"{tok['num_token_bf16_in_KVcache']} tokens")
            m3.metric("FP8", f"{tok['num_token_fp8_in_KVcache']} tokens")
            with st.expander("Raw response"):
                st.json(tok)


#  POST /plot-context-vs-memory
with tab_plot:
    st.caption("Total VRAM (weights + KV) as a function of context, per precision.")
    max_tokens = st.number_input(
        "Max context on X axis (tokens)", min_value=1, value=131072, step=1024
    )

    if st.button("Generate plot", type="primary", key="btn_plot"):
        r = requests.post(
            f"{API}/plot-context-vs-memory",
            params={"max_tokens": max_tokens},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            st.image(r.content, caption=f"{name} — context vs memory")
