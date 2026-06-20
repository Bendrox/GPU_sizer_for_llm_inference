FROM python:3.13-slim

# multi-stage build : Docker cherche l'image officielle de uv et en copie les exécutables (/uv et /uvx) 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 

# huggingface Spaces impose de tourner en utilisateur non-root (UID 1000)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# copie tout le projet > installe dépendances (sans outils dev)
COPY --chown=user . .
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["bash", "start.sh"]