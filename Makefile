.PHONY: install run format lint test

install:   
	uv sync

run-uvicorn:  
	uv run uvicorn app.app:app --reload

run-ui: 
	uv run streamlit run ui/streamlit.py

format:   
	uv run black .

lint:   
	uv run ruff check .

test:  
	uv run pytest -v

build-locally:
	docker build -t gpu-sizer .

launch-locally: 
	docker run --rm -p 7860:7860 gpu-sizer