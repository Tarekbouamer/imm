PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := imm

SRC_DIR := ./$(PROJECT_NAME)

.PHONY: install dev clean lint format check

install:
	$(PIP) install .

dev:
	$(PIP) install -e .[dev,docs,test]

torch:
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	$(PIP) uninstall -y $(PROJECT_NAME)

lint:
	ruff check $(SRC_DIR) --fix
	ruff format $(SRC_DIR)

format:
	ruff format $(SRC_DIR)

check:
	ruff check $(SRC_DIR)

test:
	$(PYTHON) -m pytest tests
