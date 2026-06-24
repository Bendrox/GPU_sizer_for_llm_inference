import altair as alt
import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"

# GPU catalog -> VRAM in GB (manufacturer 'on the box' convention, base 1000)
GPUS = {
    "NVIDIA H200 (141 GB)": 141.0,
    "NVIDIA H100 (80 GB)": 80.0,
    "NVIDIA A100 (40 GB)": 40.0,
    "NVIDIA RTX A6000 (48 GB)": 48.0,
    "NVIDIA V100 (32 GB)": 32.0,
    "NVIDIA L4 (24 GB)": 24.0,
    "NVIDIA RTX 4090 (24 GB)": 24.0,
    "NVIDIA RTX 3090 (24 GB)": 24.0,
}


def vram_donut(weights_gb: float, kv_gb: float, center: str):
    """Donut showing the model-weights vs KV-cache share of VRAM, with a centered label."""
    src = pd.DataFrame(
        {"Component": ["Model weights", "KV cache"], "GB": [weights_gb, kv_gb]}
    )
    ring = (
        alt.Chart(src)
        .mark_arc(innerRadius=55, outerRadius=90)
        .encode(
            theta=alt.Theta("GB:Q", stack=True),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(
                    domain=["Model weights", "KV cache"],
                    range=["#4C78A8", "#F58518"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Component", alt.Tooltip("GB:Q", format=".1f")],
        )
    )
    label = (
        alt.Chart(pd.DataFrame({"t": [center]}))
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(text="t:N")
    )
    return (ring + label).properties(height=240)


st.set_page_config(page_title="GPU memory Sizer", page_icon="🧮", layout="wide")
st.title("🧮 GPU memory calculator for LLM inference")


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
st.subheader("Step 1 : Select your languague model")
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

st.subheader("Step 2 : Run simulations ")

#  Tabs: one per POST endpoint
tab_kv, tab_ctx, tab_plot = st.tabs(
    ["🧠 KV cache", "📏 Max context / GPU", "📈 Context vs memory plot"]
)


#  POST /kv-cache-size-calculator
with tab_kv:
    st.caption("Calculate memory needed for a given context for a language model")
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
        # Keep the result so the chart/table survive reruns (e.g. the table toggle)
        st.session_state["kv_result"] = r.json() if r.ok else None
        if not r.ok:
            st.error(r.text)

    kv = st.session_state.get("kv_result")
    if kv:
        kv_only = kv["kv_cache_only_mb"]
        weights = kv["model_weights_mb"]
        totals = kv["totals_mb"]
        quants = ["fp32", "bf16", "fp8"]
        labels = {"fp32": "FP32", "bf16": "BF16", "fp8": "FP8"}
        order = ["FP32", "BF16", "FP8"]  # quantization increases left -> right

        if kv["includes_model_weights"]:
            rows = [
                {"Model": labels[w], "KV": labels[k], "Component": comp, "GB": gb, "rank": rank}
                for w in quants
                for k in quants
                for comp, gb, rank in (
                    ("Model weights", weights[w] / 1000, 0),
                    ("KV cache", kv_only[k] / 1000, 1),
                )
            ]
            chart = (
                alt.Chart(pd.DataFrame(rows))
                .mark_bar()
                .encode(
                    x=alt.X("Model:N", sort=order, title="Model precision  (quantization →)"),
                    xOffset=alt.XOffset("KV:N", sort=order),
                    y=alt.Y("GB:Q", title="VRAM (GB)"),
                    color=alt.Color(
                        "Component:N",
                        title="",
                        scale=alt.Scale(
                            domain=["Model weights", "KV cache"],
                            range=["#4C78A8", "#F58518"],
                        ),
                    ),
                    order=alt.Order("rank:Q"),
                    tooltip=["Model", "KV", "Component", alt.Tooltip("GB:Q", format=".2f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Per model group: 3 bars (KV FP32/BF16/FP8) · blue = weights, orange = KV cache")

            if st.toggle("Show table", key="kv_table"):
                table = {"Weights (GB)": {labels[w]: weights[w] / 1000 for w in quants}}
                for k in quants:
                    table[f"Total · KV {labels[k]} (GB)"] = {
                        labels[w]: totals[w][k] / 1000 for w in quants
                    }
                st.dataframe(pd.DataFrame(table).style.format("{:.1f}"))
        else:
            df = pd.DataFrame({"KV cache (MB)": [kv_only[k] for k in quants]}, index=order)
            st.bar_chart(df, x_label="KV cache precision", y_label="MB")
            if st.toggle("Show table", key="kv_table"):
                st.dataframe(df)

        with st.expander("Raw response"):
            st.json(kv)


#  POST /max-context-len-4-GPU-memory
with tab_ctx:
    st.caption("Number of tokens storable in your GPU (VRAM) KV cache for a given model.")

    source = st.radio(
        "VRAM Input",
        ["Input the Memory amount", "List of GPU"],
        horizontal=True,
        key="ctx_source",
    )
    c1, c2, c3 = st.columns(3)
    if source == "Input the Memory amount":
        vram = c1.number_input("Available VRAM (GB)", min_value=1.0, value=24.0, step=1.0)
    else:
        gpu = c1.selectbox("GPU", list(GPUS), key="ctx_gpu")
        vram = GPUS[gpu]
        c1.caption(f"VRAM: {vram:g} GB")
    cfg["model_quantization_bytes"] = c2.selectbox(
        "Model weight quantization",
        [4, 2, 1],
        index=[4, 2, 1].index(cfg.get("model_quantization_bytes", 2)),
        format_func=lambda v: quant_label[v],
        key="ctx_quant",
    )
    inc_weights = c3.checkbox("Include model weights", value=True, key="ctx_weights")

    if st.button("Compute max context", type="primary", key="btn_ctx"):
        r = requests.post(
            f"{API}/max-context-len-4-GPU-memory",
            params={"vram_gb": vram, "include_model_weights": inc_weights},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            tok = r.json()
            labels = {"fp32": "FP32", "bf16": "BF16", "fp8": "FP8"}
            tokens = {
                "fp32": tok["num_token_fp32_in_KVcache"],
                "bf16": tok["num_token_bf16_in_KVcache"],
                "fp8": tok["num_token_fp8_in_KVcache"],
            }

            weights_gb = (
                cfg["total_params_billion"] * cfg["model_quantization_bytes"]
                if inc_weights
                else 0
            )
            kv_gb = max(vram - weights_gb, 0)

            if not tok.get("vrm_enough_for_model"):
                st.warning("Model weights exceed the available VRAM — it does not fit.")
            st.caption(
                f"VRAM {vram:g} GB · model weights {weights_gb:g} GB · free for KV cache {kv_gb:g} GB"
            )

            # one donut per KV-cache quantization: weights vs KV share, tokens in the center
            for col, q in zip(st.columns(3), ["fp32", "bf16", "fp8"]):
                col.markdown(f"**KV {labels[q]}**")
                col.altair_chart(
                    vram_donut(weights_gb, kv_gb, tokens[q]),
                    use_container_width=True,
                )
                col.caption("max tokens in KV cache")

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
