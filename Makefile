SHELL := /bin/bash

# Load environment variables if present
-include .env
export

PROJECT_NAME := driftwatch

# Job images contain the source and configs. Refresh them before every run;
# Docker reuses cached layers when nothing changed.
RUN_JOB := docker compose --profile jobs run --build --rm

.PHONY: help up down build logs api-logs \
        gen-base gen-feature gen-blackfriday gen-card-testing \
        train promote-prod monitor control reload-api rollback \
        demo-drift-feature demo-black-friday demo-card-testing \
        clean-shared \
        format lint test check ci-local \
        setup-dev install-hooks dashboard

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
	@echo "  gen-blackfriday    Generate current dataset (shock) -> shared/data/current.csv"
	@echo "  gen-card-testing  Generate current dataset (card-testing shock) -> shared/data/current.csv"
	@echo ""
	@echo "  train              Train + register model to MLflow"
	@echo "  promote-prod       Promote latest model version to Production"
	@echo "  monitor            Run drift monitoring (Evidently) -> shared/reports/"
	@echo "  control            Run control plane (Sentinel -> Planner -> Release)"
	@echo "  rollback           Roll back to the previous Production model version"
	@echo ""
	@echo "  demo-drift-feature End-to-end demo: feature drift"
	@echo "  demo-black-friday  End-to-end demo: shock event"
	@echo "  demo-card-testing End-to-end demo: card-testing attack"
	@echo ""
	@echo "  clean-shared       Remove generated files in this project’s shared directory"
	@echo ""
	@echo "Development & CI/CD:"
	@echo "  dashboard          Open the service dashboard in browser"
	@echo "  setup-dev          Install development dependencies"
	@echo "  install-hooks      Install pre-commit hooks"
	@echo "  format             Auto-fix imports and style with ruff"
	@echo "  lint               Run linting checks (ruff, mypy)"
	@echo "  test               Run tests with coverage"
	@echo "  check              Run all quality checks (format + lint + test)"
	@echo "  ci-local           Simulate CI pipeline locally"

dashboard:
	@-[ -f .dashboard.pid ] && kill $$(cat .dashboard.pid) 2>/dev/null; rm -f .dashboard.pid
	@python3 infra/dashboard/server.py & echo $$! > .dashboard.pid
	@sleep 0.4 && open http://localhost:8765

up:
	docker compose up -d --build mlflow pushgateway prometheus grafana api
	@-[ -f .dashboard.pid ] && kill $$(cat .dashboard.pid) 2>/dev/null; rm -f .dashboard.pid
	@python3 infra/dashboard/server.py & echo $$! > .dashboard.pid
	@sleep 0.4 && open http://localhost:8765

down:
	docker compose down -v
	@-[ -f .dashboard.pid ] && kill $$(cat .dashboard.pid) 2>/dev/null; rm -f .dashboard.pid

build:
	docker compose build --no-cache api training monitoring control_plane

logs:
	docker compose logs -f

api-logs:
	docker compose logs -f api

# --- Data generation ---
gen-base:
	$(RUN_JOB) training \
	  python -m data.generator.generate --config /app/data/generator/config/base.yaml --out /app/shared/data/reference.csv

gen-feature:
	$(RUN_JOB) training \
	  python -m data.generator.generate --config /app/data/generator/config/drift_feature.yaml --out /app/shared/data/current.csv

gen-blackfriday:
	$(RUN_JOB) training \
	  python -m data.generator.generate --config /app/data/generator/config/shock_black_friday.yaml --out /app/shared/data/current.csv

gen-card-testing:
	$(RUN_JOB) training \
	  python -m data.generator.generate --config /app/data/generator/config/shock_card_testing.yaml --out /app/shared/data/current.csv

# --- Jobs (docker) ---
train:
	$(RUN_JOB) training \
	  python /app/services/training/train.py --reference /app/shared/data/reference.csv

promote-prod:
	$(RUN_JOB) training \
	  python -c "from services.training.train import promote_latest_to_production; promote_latest_to_production()"

monitor:
	$(RUN_JOB) monitoring \
	  python /app/services/monitoring/run_monitoring.py \
	    --reference /app/shared/data/reference.csv \
	    --current /app/shared/data/current.csv

control:
	$(RUN_JOB) control_plane \
	  python /app/services/control_plane/runner.py

rollback:
	$(RUN_JOB) control_plane \
	  python /app/services/control_plane/rollback.py
	$(MAKE) reload-api

# --- Demo flows ---
demo-drift-feature:
	$(MAKE) gen-base
	$(MAKE) train
	$(MAKE) promote-prod
	$(MAKE) gen-feature
	$(MAKE) monitor
	$(MAKE) control
	$(MAKE) reload-api

demo-black-friday:
	$(MAKE) gen-base
	$(MAKE) train
	$(MAKE) promote-prod
	$(MAKE) gen-blackfriday
	$(MAKE) monitor
	$(MAKE) control
	$(MAKE) reload-api

demo-card-testing:
	$(MAKE) gen-base
	$(MAKE) train
	$(MAKE) promote-prod
	$(MAKE) gen-card-testing
	$(MAKE) monitor
	$(MAKE) control
	$(MAKE) reload-api

reload-api:
	@echo "Reloading model in API container..."
	@curl -sf -X POST http://localhost:8000/reload | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Model reloaded: {d['model']} (stage: {d['stage']})\")" \
	  || { echo "  Failed to reload API model (is the API running?)"; exit 1; }

# --- Utilities ---
clean-shared:
	find shared/data shared/reports shared/events -type f -delete

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
	@echo "Auto-fixing imports and style with ruff..."
	ruff check --fix .
	@echo "Format complete!"

lint:
	@echo "Running ruff linter..."
	ruff check .
	@echo "Running type checks with mypy..."
	mypy services/ data/ --ignore-missing-imports --no-strict-optional
	@echo "Linting complete!"

test:
	@echo "Running tests with coverage..."
	pytest --cov=services --cov=data --cov-report=term-missing --cov-report=html
	@echo "Tests complete! Coverage report: htmlcov/index.html"

check: format lint test
	@echo "All quality checks passed!"

ci-local:
	@echo "Simulating CI pipeline locally..."
	@echo ""
	@echo "=== Checking import sorting ==="
	ruff check --select I .
	@echo ""
	@echo "=== Running linter ==="
	ruff check .
	@echo ""
	@echo "=== Running tests ==="
	pytest --cov=services --cov=data --cov-report=term-missing
	@echo ""
	@echo "=== Testing Docker builds ==="
	docker build -f services/api/Dockerfile -t driftwatch-api:test .
	docker build -f services/training/Dockerfile -t driftwatch-training:test .
	docker build -f services/monitoring/Dockerfile -t driftwatch-monitoring:test .
	docker build -f services/control_plane/Dockerfile -t driftwatch-control-plane:test .
	@echo ""
	@echo "CI simulation complete!"
