from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"

ABOUT = Path(__file__).parent / "about"


def with_overhead(mem_mb: float, frag: float, fixed_mb: int) -> int:
    """Estimated real allocation: int((kv_mb + weights_mb) * (1 + frag)) + fixed_overhead_mb."""
    return int(mem_mb * (1 + frag)) + fixed_mb


def overhead_controls(key: str):
    """Checkbox + params for the CUDA overhead estimate. Returns (enabled, frag, fixed_mb)."""
    on = st.checkbox("Add estimated CUDA overhead", key=f"ovh_{key}")
    frag, fixed = 0.0, 0
    if on:
        a, b = st.columns(2)
        frag = a.number_input("Fragmentation factor", 0.0, 1.0, 0.07, 0.01, key=f"frag_{key}")
        fixed = b.number_input("Fixed overhead (MB)", 0, 16000, 750, 50, key=f"fix_{key}")
        st.caption("real_mb = `int((kv_mb + weights_mb) * (1 + frag)) + fixed_overhead_mb`")
    return on, float(frag), int(fixed)


def vram_donut(weights_gb: float, kv_gb: float, center: str, overhead_gb: float = 0.0):
    """Donut: model-weights vs KV-cache (+ optional CUDA overhead) share of VRAM."""
    comps = [("Model weights", weights_gb), ("KV cache", kv_gb)]
    if overhead_gb > 0:
        comps.append(("CUDA overhead", overhead_gb))
    src = pd.DataFrame({"Component": [c for c, _ in comps], "GB": [g for _, g in comps]})
    ring = (
        alt.Chart(src)
        .mark_arc(innerRadius=55, outerRadius=90)
        .encode(
            theta=alt.Theta("GB:Q", stack=True),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(
                    domain=["Model weights", "KV cache", "CUDA overhead"],
                    range=["#4C78A8", "#F58518", "#B0B0B0"],
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


def about(feature: str, what_title: str):
    """Two homogeneous help sections per feature: the concept, then the technical detail."""
    with st.expander(f"{what_title}"):
        st.markdown((ABOUT / f"{feature}_concept.md").read_text(encoding="utf-8"))
    with st.expander("Technical implementation"):
        st.markdown((ABOUT / f"{feature}_tech.md").read_text(encoding="utf-8"))


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

#  GPU catalog: GET /gpus (served by the backend, single source of truth)
GPUS = requests.get(f"{API}/gpus", timeout=5).json()


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

st.subheader("Step 2 : Run a simulation")

#  Tabs: one per POST endpoint
tab_kv, tab_ctx, tab_plot, tab_vllm = st.tabs(
    ["KV cache", "Max context per GPU", "Context vs memory plot", "vLLM simulation"]
)


#  POST /kv-cache-size-calculator
with tab_kv:
    st.caption("Calculate memory needed for a given context for a language model")
    about("kv", "What is the KV cache?")

    c1, c2, c3 = st.columns(3)
    ctx = c1.number_input("Context (tokens)", min_value=1, value=8192, step=1024)
    batch = c2.number_input("Batch size", min_value=1, value=1)
    inc_weights = c3.checkbox("Include model weights", value=True)
    ovh, frag, fixed = overhead_controls("kv")

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
            rows = []
            for w in quants:
                for k in quants:
                    base = weights[w] + kv_only[k]
                    parts = [("Model weights", weights[w], 0), ("KV cache", kv_only[k], 1)]
                    if ovh:
                        parts.append(("CUDA overhead", with_overhead(base, frag, fixed) - base, 2))
                    rows += [
                        {"Model": labels[w], "KV": labels[k], "Component": c, "GB": mb / 1000, "rank": rk}
                        for c, mb, rk in parts
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
                            domain=["Model weights", "KV cache", "CUDA overhead"],
                            range=["#4C78A8", "#F58518", "#B0B0B0"],
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
                        labels[w]: (
                            with_overhead(weights[w] + kv_only[k], frag, fixed)
                            if ovh
                            else totals[w][k]
                        )
                        / 1000
                        for w in quants
                    }
                st.dataframe(pd.DataFrame(table).style.format("{:.1f}"))
        else:
            df = pd.DataFrame({"KV cache (MB)": [kv_only[k] for k in quants]}, index=order)
            if ovh:
                df["+ CUDA overhead (MB)"] = [with_overhead(kv_only[k], frag, fixed) for k in quants]
            st.bar_chart(df, x_label="KV cache precision", y_label="MB")
            if st.toggle("Show table", key="kv_table"):
                st.dataframe(df)

        with st.expander("Raw response"):
            st.json(kv)


#  POST /max-context-len-4-GPU-memory
with tab_ctx:
    st.caption("Number of tokens storable in your GPU (VRAM) KV cache for a given model.")
    about("ctx", "What is max context per GPU sizing?")

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
    ovh, frag, fixed = overhead_controls("ctx")

    if st.button("Compute max context", type="primary", key="btn_ctx"):
        # CUDA overhead reserves VRAM, so the budget for weights + KV shrinks
        vram_budget = max(vram * 1000 - fixed, 0) / (1 + frag) / 1000
        r = requests.post(
            f"{API}/max-context-len-4-GPU-memory",
            params={"vram_gb": vram_budget, "include_model_weights": inc_weights},
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
            overhead_gb = vram - vram_budget
            kv_gb = max(vram_budget - weights_gb, 0)

            if not tok.get("vrm_enough_for_model"):
                st.warning("Model weights exceed the available VRAM — it does not fit.")
            st.caption(
                f"VRAM {vram:g} GB · weights {weights_gb:g} GB"
                + (f" · CUDA overhead {overhead_gb:.1f} GB" if ovh else "")
                + f" · free for KV cache {kv_gb:.1f} GB"
            )

            # one donut per KV-cache quantization: weights vs KV share, tokens in the center
            for col, q in zip(st.columns(3), ["fp32", "bf16", "fp8"]):
                col.markdown(f"**KV {labels[q]}**")
                col.altair_chart(
                    vram_donut(weights_gb, kv_gb, tokens[q], overhead_gb if ovh else 0.0),
                    use_container_width=True,
                )
                col.caption("max tokens in KV cache")

            with st.expander("Raw response"):
                st.json(tok)


#  POST /plot-context-vs-memory
with tab_plot:
    st.caption("Total VRAM (weights + KV) as a function of context, per precision.")
    about("plot", "What is the context-vs-memory plot?")
    max_tokens = st.number_input(
        "Max context on X axis (tokens)", min_value=1, value=131072, step=1024
    )
    ovh, frag, fixed = overhead_controls("plot")

    if st.button("Generate plot", type="primary", key="btn_plot"):
        r = requests.post(
            f"{API}/plot-context-vs-memory",
            params={"max_tokens": max_tokens, "frag": frag, "fixed_overhead_mb": fixed},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            st.image(r.content, caption=f"{name} — context vs memory")


#  POST /vllm-capacity
with tab_vllm:
    st.caption("vLLM-style capacity: usable VRAM = total × utilization, KV cache paged in blocks of 16 tokens.")
    about("vllm", "What is vLLM?")

    source = st.radio(
        "VRAM Input", ["Input the Memory amount", "List of GPU"], horizontal=True, key="vllm_source"
    )
    c1, c2, c3 = st.columns(3)
    if source == "Input the Memory amount":
        vram = c1.number_input("Total VRAM (GB)", min_value=1.0, value=80.0, step=1.0, key="vllm_vram")
    else:
        gpu = c1.selectbox("GPU", list(GPUS), key="vllm_gpu")
        vram = GPUS[gpu]
        c1.caption(f"VRAM: {vram:g} GB")
    util = c2.slider("GPU memory utilization", 0.10, 1.00, 0.90, 0.05, key="vllm_util")
    seq_len = c3.number_input("Sequence length per request (tokens)", min_value=1, value=8192, step=512, key="vllm_seq")

    c4, c5 = st.columns(2)
    cfg["model_quantization_bytes"] = c4.selectbox(
        "Model weight quantization",
        [4, 2, 1],
        index=[4, 2, 1].index(cfg.get("model_quantization_bytes", 2)),
        format_func=lambda v: quant_label[v],
        key="vllm_quant",
    )
    kv_dtype = c5.selectbox(
        "KV cache dtype",
        [4, 2, 1],
        index=[4, 2, 1].index(cfg["model_quantization_bytes"]),
        format_func=lambda v: quant_label[v],
        key="vllm_kv_dtype",
    )

    if st.button("Run vLLM simulation", type="primary", key="btn_vllm"):
        r = requests.post(
            f"{API}/vllm-capacity",
            params={
                "total_vram_gb": vram,
                "seq_len": seq_len,
                "gpu_memory_utilization": util,
                "kv_dtype_bytes": kv_dtype,
            },
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            v = r.json()
            if not v["fits"]:
                st.error(
                    f"Model weights ({v['weights_mb'] / 1000:.1f} GB) don't fit in usable VRAM !"
                    f"({v['usable_vram_mb'] / 1000:.1f} GB = {util:.0%} of {vram:g} GB). "
                    "Pick a bigger GPU, raise utilization, or quantize the model further."
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Concurrent requests", f"{v['max_concurrent_requests']:,}", help=f"at seq_len = {seq_len:,}")
                m2.metric("KV blocks (×16 tok)", f"{v['num_blocks']:,}")
                m3.metric("Total KV tokens", f"{v['total_tokens']:,}")
                st.caption(
                    f"Model weights {quant_label[cfg['model_quantization_bytes']]} · "
                    f"KV cache dtype {quant_label[kv_dtype]}"
                )
                st.caption(
                    f"Usable VRAM {v['usable_vram_mb'] / 1000:.1f} GB ({util:.0%} of {vram:g} GB) · "
                    f"weights {v['weights_mb'] / 1000:.1f} GB · KV cache {v['kv_cache_mb'] / 1000:.1f} GB · "
                    f"{v['blocks_per_request']} blocks/request"
                )
            with st.expander("Raw response"):
                st.json(v)
