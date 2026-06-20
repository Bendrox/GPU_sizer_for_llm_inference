
# Script lancement des services : API (Uvicorn) & Interface (Streamlit)

# Démarre l'API FastAPI en arrière-plan, puis lance le Streamlit au premier plan pour gérer l'interface utilisateur

# 1. Lancement de l'API (FastAPI via Uvicorn)
# - '--host 0.0.0.0' : Rend l'API accessible depuis l'extérieur du conteneur/hôte.
# & Lance le processus en arrière-plan pour permettre au script de continuer.
echo "[INFO] Démarrage de l'API Uvicorn en arriere plan sur le port 8000..."
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000 &


# 2. Lancement de l'Interface Utilisateur (Streamlit)
# - 'exec' : Remplace le processus du script shell par celui de Streamlit.
#            Permet de transférer proprement les signaux (ex: SIGTERM pour l'arrêt).
#            for Graceful Shutdown 
# server.headless true Désactive l'ouverture automatique du navigateur
echo "[INFO] Démarrage de l'application Streamlit sur le port 7860..."
exec uv run streamlit run ui/streamlit.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true