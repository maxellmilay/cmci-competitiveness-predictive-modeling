# CMCI ML Pipeline Makefile

.PHONY: help install setup clean test lint format run-pipeline run-prefect-pipeline mlflow-ui

# Default target
help:
	@echo "Available commands:"
	@echo "  install          - Install dependencies"
	@echo "  setup            - Setup project (install + create directories)"
	@echo "  clean            - Clean generated files"
	@echo "  test             - Run tests"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  run-pipeline     - Run basic ML pipeline"
	@echo "  run-prefect      - Run Prefect-based pipeline"
	@echo "  mlflow-ui        - Start MLflow UI"
	@echo "  jupyter          - Start Jupyter server"

# Installation and setup
install:
	pip install -r requirements.txt

setup: install
	mkdir -p data/raw data/processed data/features
	mkdir -p models/artifacts
	mkdir -p logs metrics plots
	mkdir -p config

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf mlruns/
	rm -rf logs/*.log
	rm -rf plots/*.png

# Development
test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ --max-line-length=100
	black --check src/

format:
	black src/
	isort src/

# Pipeline execution
run-pipeline:
	python -m src.pipeline.orchestrator

run-prefect:
	python -m src.pipeline.prefect_pipeline

# MLflow UI
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Jupyter
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Data validation
validate-data:
	python -c "from src.data.validation import validate_all_data; validate_all_data()"

# Model comparison
compare-models:
	python -c "from src.visualization.evaluation import compare_all_models; compare_all_models()"
