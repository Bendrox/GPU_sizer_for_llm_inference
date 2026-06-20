.PHONY: install run format lint test

install:   
	uv sync

run-uvicorn:  
	uv run uvicorn app.app:app --reload

run-fr-ui: 
	uv run streamlit run streamlit_app_fr.py

run-eng-ui: 
	uv run streamlit run streamlit_app_eng.py

format:   
	uv run black .

lint:   
	uv run ruff check .

test:  
	uv run pytest -v