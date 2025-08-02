# PDF Summarizer - Makefile
# Professional development automation

.PHONY: help install dev prod test clean lint format run docker-build docker-run docs

# Default target
help:
	@echo "PDF Summarizer - Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install     - Install all dependencies"
	@echo "  make dev         - Setup development environment"
	@echo "  make prod        - Setup production environment"
	@echo ""
	@echo "Development Commands:"
	@echo "  make run         - Start development server"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with black"
	@echo "  make clean       - Clean temporary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs        - Generate documentation"
	@echo ""

# Installation targets
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

dev: install
	@echo "Setting up development environment..."
	pip install -r requirements-dev.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
	@echo "Development environment ready"

prod: install
	@echo "Setting up production environment..."
	@echo "Production environment ready"

# Development targets
run:
	@echo "Starting development server..."
	python run.py --env development --debug

run-prod:
	@echo "Starting production server..."
	python run.py --env production --no-debug

test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=core --cov=src --cov=utils
	@echo "Tests completed"

test-api:
	@echo "Testing API endpoints..."
	python -m pytest tests/test_api.py -v
	@echo "API tests completed"

lint:
	@echo "Running code linting..."
	flake8 . --max-line-length=120 --exclude=venv,env,models,data
	pylint core/ src/ utils/ app.py config.py run.py
	@echo "Linting completed"

format:
	@echo "Formatting code..."
	black . --line-length=120 --exclude="/(\.git|\.mypy_cache|\.tox|\.venv|_build|buck-out|build|dist|models|data)/"
	isort . --profile black
	@echo "Code formatted"

# Cleaning targets
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	@echo "Cleanup completed"

clean-models:
	@echo "Cleaning model cache..."
	rm -rf models/
	rm -rf ~/.cache/huggingface/
	@echo "Model cache cleaned"

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t pdf-summarizer .
	@echo "Docker image built"

docker-run:
	@echo "Running Docker container..."
	docker run -p 5000:5000 pdf-summarizer
	@echo "Docker container started"

docker-dev:
	@echo "Running Docker in development mode..."
	docker run -p 5000:5000 -v $(PWD):/app pdf-summarizer
	@echo "Docker development container started"

# Documentation targets
docs:
	@echo "Generating documentation..."
	mkdir -p docs
	python -c "
import sys
sys.path.append('.')
from core.generator import generate_summary
from src.translator_models import Translator
help(generate_summary)
help(Translator)
" > docs/api_reference.txt
	@echo "Documentation generated in docs/"

# Quality assurance
qa: lint test
	@echo "Quality assurance completed"

# Security check
security:
	@echo "Running security checks..."
	pip install bandit safety
	bandit -r . -f json -o security_report.json
	safety check
	@echo "Security check completed"

# Performance profiling
profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile_stats.prof app.py
	@echo "Profiling completed - check profile_stats.prof"

# Database/Model management
download-models:
	@echo "Downloading AI models..."
	python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
print('Downloading BART model...')
AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
print('Downloading translation models...')
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE')
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ROMANCE')
print('All models downloaded successfully!')
"
	@echo "Models downloaded"

# Environment setup
env-create:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created"
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

env-activate:
	@echo "Please run: source venv/bin/activate"

# Git helpers
git-setup:
	@echo "Setting up git hooks..."
	echo "#!/bin/bash\nmake lint" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Git hooks configured"

# Comprehensive setup
setup: env-create install dev download-models
	@echo "Complete setup finished!"
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Start development server: make run"

# Health check
health:
	@echo "Performing health check..."
	python -c "
import requests
import time
import subprocess
import signal
import os

# Start server in background
proc = subprocess.Popen(['python', 'run.py', '--skip-models'], 
						stdout=subprocess.DEVNULL, 
						stderr=subprocess.DEVNULL)
time.sleep(3)

try:
    response = requests.get('http://localhost:5000/')
    if response.status_code == 200:
        print('API server is healthy')
    else:
        print('API server returned error:', response.status_code)
except:
    print('API server is not responding')
finally:
    proc.terminate()
    proc.wait()
"

# All-in-one commands
all: clean install lint test
	@echo "All tasks completed successfully"

deploy-check: lint test security
	@echo "Deployment checks passed"
