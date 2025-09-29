import requests
import streamlit as st

API = "http://localhost:8000"  

st.set_page_config(page_title="GPU Sizer", page_icon="🧮")
st.title("GPU Sizer pour inférence LLM")

models = requests.get(f"{API}/models").json()
nom = st.selectbox("Modèle", models)

ctx = st.number_input("Longueur de contexte (tokens)", min_value=1, value=8192, step=1024)
batch = st.number_input("Batch size (requêtes simultanées)", min_value=1, value=1)

if st.button("Calculer", type="primary"):
    # récupère la config complète du modèle choisi
    config = requests.get(f"{API}/models/{nom}").json()

    kv = requests.post(
        f"{API}/kv-cache-size-calculator",
        params={"length_seq": ctx, "batch_size": batch, "include_model_weights": True},
        json=config,
    ).json()

    st.subheader("Mémoire totale requise (poids + KV cache)")
    c1, c2, c3 = st.columns(3)
    c1.metric("FP32", f"{kv['memory_consumption_fp32_mo']:,} Mo")
    c2.metric("BF16", f"{kv['memory_consumption_bf16_mo']:,} Mo")
    c3.metric("FP8",  f"{kv['memory_consumption_fp8_mo']:,} Mo")

    with st.expander("Réponse brute de l'API"):
        st.json(kv)