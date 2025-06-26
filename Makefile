.PHONY: help install install-dev test test-cov lint format type-check clean build docs

# Default target
help:
	@echo "QuarryCore Development Makefile"
	@echo "Production-grade AI training data miner with adaptive hardware optimization"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install           Basic installation"
	@echo "  install-dev       Development environment"
	@echo "  install-gpu       GPU acceleration (workstation)"
	@echo "  install-pi        Raspberry Pi optimization"
	@echo "  install-workstation  Complete workstation setup"
	@echo "  install-research  Research environment (all domains)"
	@echo "  install-medical   Medical domain processing"
	@echo "  install-legal     Legal domain processing"
	@echo "  install-all       Complete installation"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test              Run all tests"
	@echo "  test-cov          Run with coverage report"
	@echo "  test-gpu          GPU-specific tests"
	@echo "  test-integration  Integration tests"
	@echo "  test-unit         Unit tests only"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  lint              Run flake8 linting"
	@echo "  format            Format with black and isort"
	@echo "  type-check        Run mypy type checking"
	@echo "  pre-commit        Run all pre-commit hooks"
	@echo ""
	@echo "🏗️  Build:"
	@echo "  clean             Clean build artifacts"
	@echo "  build             Build the package"
	@echo "  docs              Build documentation"

# Installation (adaptive hardware support)
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pre-commit install

# Hardware-specific installations (following workflow specs)
install-gpu:
	pip install -e .[gpu,performance]
	@echo "GPU acceleration enabled for workstation mode"

install-pi:
	pip install -e .[pi,performance]
	@echo "Raspberry Pi optimization enabled"

install-workstation:
	pip install -e .[workstation]
	@echo "Full workstation setup with GPU acceleration"

install-research:
	pip install -e .[research]
	@echo "Research environment with all domain processors"

# Domain-specific installations
install-medical:
	pip install -e .[medical]
	python -m spacy download en_core_web_sm

install-legal:
	pip install -e .[legal]

install-all:
	pip install -e .[all]
	python -m spacy download en_core_web_sm
	@echo "Complete QuarryCore installation with all features"

# Testing
test:
	pytest -v

test-cov:
	pytest --cov=quarrycore --cov-report=html --cov-report=term

test-gpu:
	pytest -m gpu

test-integration:
	pytest -m integration

test-unit:
	pytest -m unit

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

pre-commit:
	pre-commit run --all-files

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:
	python -m build

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

docs-serve:
	sphinx-autobuild docs/ docs/_build/

# Development
dev-setup: install-dev
	@echo "Development environment set up!"

# Quick quality check
check: format lint type-check test

# Release preparation
release-check: clean check test-cov
	@echo "Release ready!" 