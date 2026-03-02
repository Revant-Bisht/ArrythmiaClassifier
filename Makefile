.PHONY: help install install-dev setup-hooks download-data process-data \
        train evaluate export-onnx cache-samples serve test lint format clean

PYTHON := python
PIP    := pip
SRC    := src
CONFIG := configs/default.yaml

help:
	@echo ""
	@echo "Arrhythmia Classifier — available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install core dependencies"
	@echo "    make install-dev    Install core + dev/lint/test dependencies"
	@echo "    make setup-hooks    Install pre-commit hooks"
	@echo ""
	@echo "  Data"
	@echo "    make download-data  Download PTB-XL from PhysioNet"
	@echo "    make process-data   Pre-process raw ECG records into tensors"
	@echo ""
	@echo "  Training"
	@echo "    make train          Train the InceptionTime+Attention model"
	@echo "    make evaluate       Run evaluation on the held-out test set"
	@echo "    make export-onnx    Export trained model to ONNX"
	@echo "    make cache-samples  Pre-compute 5 sample responses for the API cache"
	@echo "    make serve          Run FastAPI backend locally on port 8080"
	@echo ""
	@echo "  Quality"
	@echo "    make test           Run pytest with coverage"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format with black + ruff --fix"
	@echo ""
	@echo "    make clean          Remove caches and compiled artefacts"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

setup-hooks:
	pre-commit install

download-data:
	$(PYTHON) scripts/download_ptbxl.py --output-dir data/raw/ptb-xl

process-data:
	$(PYTHON) scripts/process_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

export-onnx:
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG)

cache-samples:
	$(PYTHON) scripts/cache_samples.py --config $(CONFIG)

serve:
	uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest

lint:
	ruff check $(SRC) tests scripts
	mypy $(SRC)

format:
	black $(SRC) tests scripts
	ruff check --fix $(SRC) tests scripts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache
