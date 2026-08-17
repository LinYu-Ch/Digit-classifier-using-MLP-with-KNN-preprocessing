.PHONY: help install test lint format clean

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make install   Create a virtual environment and install dependencies"
	@echo "  make test      Run the test suite"
	@echo "  make lint      Run lint checks"
	@echo "  make format    Format the codebase"
	@echo "  make clean     Remove generated artifacts"
	@echo "  make train     Run the existing training loop"
	@echo "  make run       Execute the inference model"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	if [ -f requirements.txt ]; then $(VENV_PIP) install -r requirements.txt; fi
	if [ -f pyproject.toml ]; then $(VENV_PIP) install -e .; fi

test:
	if [ -f $(VENV_PYTHON) ]; then $(VENV_PYTHON) -m pytest; else $(PYTHON) -m pytest; fi

lint:
	if [ -f $(VENV_PYTHON) ]; then $(VENV_PYTHON) -m ruff check .; else $(PYTHON) -m ruff check .; fi

format:
	if [ -f $(VENV_PYTHON) ]; then $(VENV_PYTHON) -m ruff format .; else $(PYTHON) -m ruff format .; fi

clean:
	rm -rf .pytest_cache .ruff_cache .coverage .mypy_cache __pycache__ build dist *.egg-info $(VENV)

train:
	$(PYTHON) train.py

run:
	$(PYTHON) classify.py