.PHONY: install run format lint test

install:   
	uv sync

run-swagger:  
	uv run uvicorn app.app:app --reload

run-ui: 
	uv run streamlit run streamlit_app.py

format:   
	uv run black .

lint:   
	uv run ruff check .

test:  
	uv run pytest