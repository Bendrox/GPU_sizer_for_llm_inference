import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="GPU Sizer", page_icon="🧮", layout="wide")
st.title("🧮 GPU Sizer pour inférence LLM")


#  Sidebar 
with st.sidebar:
    st.header("⚙️ API")
    API = st.text_input("URL de l'API", API)

    if st.button("Ping /health", use_container_width=True):
        try:
            r = requests.get(f"{API}/health", timeout=5)
            st.success(r.json()) if r.ok else st.error(r.text)
        except requests.RequestException as e:
            st.error(f"Injoignable : {e}")


#  Chargement du catalogue 
try:
    models = requests.get(f"{API}/models", timeout=5).json()
except requests.RequestException as e:
    st.error(f"Impossible de charger les modèles depuis {API} : {e}")
    st.stop()


# Choix du modèle + config éditable
st.subheader("Modèle")
col_sel, col_quant = st.columns([2, 1])

nom = col_sel.selectbox("Catalogue", models)
cfg = requests.get(f"{API}/models/{nom}").json()

with st.expander("Architecture du modèle (éditable)", expanded=False):
    c1, c2, c3 = st.columns(3)
    cfg["nbr_attention_heads_kv"] = c1.number_input(
        "Têtes d'attention KV", min_value=1, value=cfg["nbr_attention_heads_kv"])
    cfg["nbr_head_dim"] = c2.number_input(
        "Dimension par tête", min_value=1, value=cfg["nbr_head_dim"])
    cfg["nbr_Gated_Attention_layers"] = c3.number_input(
        "Couches Gated Attention", min_value=1, value=cfg["nbr_Gated_Attention_layers"])

    c4, c5 = st.columns(2)
    cfg["total_params_billion"] = c4.number_input(
        "Paramètres (milliards)", min_value=1, value=cfg["total_params_billion"])
    quant_label = {4: "FP32 (4)", 2: "BF16 (2)", 1: "FP8 (1)"}
    cfg["model_quantization_oct"] = c5.selectbox(
        "Quantification des poids", [4, 2, 1],
        index=[4, 2, 1].index(cfg.get("model_quantization_oct", 2)),
        format_func=lambda v: quant_label[v])

    st.json(cfg)


tab_kv, tab_ctx, tab_plot = st.tabs(
    ["KV cache", "Contexte max / GPU", "📈 Graphe contexte vs mémoire"])


with tab_kv:
    st.caption("Mémoire pour un contexte donné (option : + poids du modèle, batch).")
    c1, c2, c3 = st.columns(3)
    ctx = c1.number_input("Contexte (tokens)", min_value=1, value=8192, step=1024)
    batch = c2.number_input("Batch size", min_value=1, value=1)
    inc_weights = c3.checkbox("Inclure les poids du modèle", value=True)

    if st.button("Calculer le KV cache", type="primary", key="btn_kv"):
        r = requests.post(
            f"{API}/kv-cache-size-calculator",
            params={"length_seq": ctx, "batch_size": batch,
                    "include_model_weights": inc_weights},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            kv = r.json()
            label = "totale (poids + KV)" if inc_weights else "KV cache seul"
            st.markdown(f"**Mémoire {label}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("FP32", f"{kv['memory_consumption_fp32_mo']:,} Mo")
            m2.metric("BF16", f"{kv['memory_consumption_bf16_mo']:,} Mo")
            m3.metric("FP8", f"{kv['memory_consumption_fp8_mo']:,} Mo")

            if kv["includes_model_weights"]:
                st.caption(f"Poids du modèle : {kv['model_weights_mo']:,} Mo")
            with st.expander("Réponse brute"):
                st.json(kv)


with tab_ctx:
    st.caption("Nombre de tokens stockables en KV cache pour une VRAM donnée.")
    vram = st.number_input("VRAM disponible (Go)", min_value=1.0, value=24.0, step=1.0)

    if st.button("Calculer le contexte max", type="primary", key="btn_ctx"):
        r = requests.post(
            f"{API}/max-context-len-4-GPU-memory",
            params={"vram_go": vram},
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
            with st.expander("Réponse brute"):
                st.json(tok)


with tab_plot:
    st.caption("VRAM totale (poids + KV) en fonction du contexte, par précision.")
    max_tokens = st.number_input(
        "Contexte max sur l'axe X (tokens)", min_value=1, value=131072, step=1024)

    if st.button("Générer le graphe", type="primary", key="btn_plot"):
        r = requests.post(
            f"{API}/plot-context-vs-memory",
            params={"max_tokens": max_tokens},
            json=cfg,
        )
        if not r.ok:
            st.error(r.text)
        else:
            st.image(r.content, caption=f"{nom} — contexte vs mémoire")
