.PHONY: install run format lint test

install:   ## Installe les dépendances
	uv sync

run:       ## Lance l'API (rechargement auto)
	uv run uvicorn app.app:app --reload

format:    ## Formate avec black
	uv run black .

lint:      ## Analyse avec ruff
	uv run ruff check .

test:      ## Lance les tests
	uv run pytest