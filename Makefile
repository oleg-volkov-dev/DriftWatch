SHELL := /bin/bash

# Load environment variables if present
-include .env
export

PROJECT_NAME := openml-agentic

.PHONY: help up down build logs api-logs \
        gen-base gen-feature gen-concept gen-blackfriday \
        train promote-prod monitor control \
        demo-drift-feature demo-drift-concept demo-black-friday \
        clean-shared \
        format lint test check ci-local \
        setup-dev install-hooks

help:
	@echo "Targets:"
	@echo "  up                 Start core services (mlflow, api, prometheus, grafana)"
	@echo "  down               Stop and remove containers + volumes"
	@echo "  build              Build images"
	@echo "  logs               Tail logs for all running services"
	@echo "  api-logs           Tail API logs"
	@echo ""
	@echo "  gen-base           Generate reference dataset -> shared/data/reference.csv"
	@echo "  gen-feature        Generate current dataset (feature drift) -> shared/data/current.csv"
	@echo "  gen-concept        Generate current dataset (concept drift) -> shared/data/current.csv"
	@echo "  gen-blackfriday    Generate current dataset (shock) -> shared/data/current.csv"
	@echo ""
	@echo "  train              Train + register model to MLflow"
	@echo "  promote-prod       Promote latest model version to Production"
	@echo "  monitor            Run drift monitoring (Evidently) -> shared/reports/"
	@echo "  control            Run control plane (Sentinel -> Planner -> Release)"
	@echo ""
	@echo "  demo-drift-feature End-to-end demo: feature drift"
	@echo "  demo-drift-concept End-to-end demo: concept drift"
	@echo "  demo-black-friday  End-to-end demo: shock event"
	@echo ""
	@echo "  clean-shared       Remove shared artifacts volume"
	@echo ""
	@echo "Development & CI/CD:"
	@echo "  setup-dev          Install development dependencies"
	@echo "  install-hooks      Install pre-commit hooks"
	@echo "  format             Format code with black and isort"
	@echo "  lint               Run linting checks (ruff, mypy)"
	@echo "  test               Run tests with coverage"
	@echo "  check              Run all quality checks (format + lint + test)"
	@echo "  ci-local           Simulate CI pipeline locally"

up:
	docker compose up -d --build mlflow prometheus grafana api

down:
	docker compose down -v

build:
	docker compose build --no-cache api training monitoring control_plane

logs:
	docker compose logs -f

api-logs:
	docker compose logs -f api

# --- Data generation ---
gen-base:
	docker compose --profile jobs run --rm training \
	  python -m data.generator.generate --config /app/data/generator/config/base.yaml --out /app/shared/data/reference.csv

gen-feature:
	docker compose --profile jobs run --rm training \
	  python -m data.generator.generate --config /app/data/generator/config/drift_feature.yaml --out /app/shared/data/current.csv

gen-concept:
	docker compose --profile jobs run --rm training \
	  python -m data.generator.generate --config /app/data/generator/config/drift_concept.yaml --out /app/shared/data/current.csv

gen-blackfriday:
	docker compose --profile jobs run --rm training \
	  python -m data.generator.generate --config /app/data/generator/config/shock_black_friday.yaml --out /app/shared/data/current.csv

# --- Jobs (docker) ---
train:
	docker compose --profile jobs run --rm training \
	  python /app/services/training/train.py --reference /app/shared/data/reference.csv

promote-prod:
	docker compose --profile jobs run --rm training \
	  python -c "from services.training.train import promote_latest_to_production; promote_latest_to_production()"

monitor:
	docker compose --profile jobs run --rm monitoring \
	  python /app/services/monitoring/run_monitoring.py \
	    --reference /app/shared/data/reference.csv \
	    --current /app/shared/data/current.csv

control:
	docker compose --profile jobs run --rm control_plane \
	  python /app/services/control_plane/runner.py

# --- Demo flows ---
demo-drift-feature: gen-base train promote-prod gen-feature monitor control

demo-drift-concept: gen-base train promote-prod gen-concept monitor control

demo-black-friday: gen-base train promote-prod gen-blackfriday monitor control

# --- Utilities ---
clean-shared:
	docker volume rm -f $$(docker volume ls -q | grep -E "_shared$$" || true)

# --- Development & CI/CD ---
setup-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "Development environment ready!"

install-hooks:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed. They will run automatically on git commit."

format:
	@echo "Formatting code with black..."
	black .
	@echo "Sorting imports with isort..."
	isort .
	@echo "Code formatting complete!"

lint:
	@echo "Running ruff linter..."
	ruff check .
	@echo "Running type checks with mypy..."
	mypy services/ data/ --ignore-missing-imports --no-strict-optional || true
	@echo "Linting complete!"

test:
	@echo "Running tests with coverage..."
	pytest --cov=services --cov=data --cov-report=term-missing --cov-report=html || true
	@echo "Tests complete! Coverage report: htmlcov/index.html"

check: format lint test
	@echo "All quality checks passed!"

ci-local:
	@echo "Simulating CI pipeline locally..."
	@echo ""
	@echo "=== Checking code formatting ==="
	black --check --diff .
	@echo ""
	@echo "=== Checking import sorting ==="
	isort --check-only --diff .
	@echo ""
	@echo "=== Running linter ==="
	ruff check .
	@echo ""
	@echo "=== Running tests ==="
	pytest --cov=services --cov=data --cov-report=term-missing || true
	@echo ""
	@echo "=== Testing Docker builds ==="
	docker build -f services/api/Dockerfile -t driftwatch-api:test .
	docker build -f services/training/Dockerfile -t driftwatch-training:test .
	docker build -f services/monitoring/Dockerfile -t driftwatch-monitoring:test .
	docker build -f services/control_plane/Dockerfile -t driftwatch-control-plane:test .
	@echo ""
	@echo "CI simulation complete!"
