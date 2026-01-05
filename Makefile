SHELL := /bin/bash

# Load environment variables if present
-include .env
export

PROJECT_NAME := openml-agentic

.PHONY: help up down build logs api-logs \
        gen-base gen-feature gen-concept gen-blackfriday \
        train promote-prod monitor control \
        demo-drift-feature demo-drift-concept demo-black-friday \
        clean-shared

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
