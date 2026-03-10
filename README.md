# DriftWatch

[![CI](https://github.com/oleg-volkov-dev/DriftWatch/actions/workflows/ci.yml/badge.svg)](https://github.com/oleg-volkov-dev/DriftWatch/actions/workflows/ci.yml)
[![Docker Build](https://github.com/oleg-volkov-dev/DriftWatch/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/oleg-volkov-dev/DriftWatch/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Autonomous MLOps platform for fraud detection with policy-driven drift response**

A production-grade MLOps demonstration that automatically detects, evaluates, and responds to model drift using a multi-agent control plane. Built with synthetic fraud data to showcase real-world ML reliability patterns.

---

## Overview

DriftWatch simulates a complete ML lifecycle where models monitor themselves and autonomously decide when to retrain, deploy, or rollback based on drift detection and quality gates.

### Key Features

- **Web Dashboard**: Browser UI at `localhost:8765` — run pipeline steps, launch demos, inspect the live model, and call the inference API without touching the terminal
- **Automated Drift Detection**: Monitors feature distribution drift using Evidently
- **Multi-Agent Control Plane**: Three specialized agents coordinate automated responses
  - **Sentinel**: Observes metrics and classifies incidents
  - **Planner**: Generates execution plans based on policies
  - **Release**: Enforces quality gates and safe deployments
- **Policy-as-Code**: All decisions constrained by versioned YAML policies
- **Full Observability**: MLflow tracking, Prometheus metrics, drift reports
- **Reproducible Demo Scenarios**: Pre-configured drift scenarios for testing

---

## Architecture

```
┌─────────────────┐
│  Synthetic Data │  ← Configurable drift injection
│    Generator    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Training       │────▶│   MLflow     │
│  Pipeline       │     │   Registry   │
└─────────────────┘     └──────┬───────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  FastAPI Service │  ← Serves predictions
                    └──────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌──────────────┐    ┌──────────────┐
│   Monitoring    │   │  Prometheus  │    │   Grafana    │
│   (Evidently)   │   │   Metrics    │    │  Dashboard   │
└────────┬────────┘   └──────────────┘    └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Control Plane (Agents)                     │
│  ┌──────────┐   ┌─────────┐   ┌───────────────────┐   │
│  │ Sentinel │──▶│ Planner │──▶│ Release Agent     │   │
│  │ (Detect) │   │(Decide) │   │(Execute Safely)   │   │
│  └──────────┘   └─────────┘   └───────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **ML**: scikit-learn, MLflow
- **API**: FastAPI, Uvicorn
- **Monitoring**: Evidently (drift), Prometheus, Grafana
- **Infrastructure**: Docker Compose
- **Language**: Python 3.11+

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for data generation)
- `pip install pyyaml numpy pandas`

### 1. Start Services

```bash
make up
```

`make up` builds and starts all containers, then automatically opens the **DriftWatch dashboard** at `http://localhost:8765`. Everything else can be done from there.

Direct service links:
- **Dashboard**: http://localhost:8765
- **MLflow**: http://localhost:5000
- **API docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### 2. Run a Complete Demo

```bash
make demo-drift-feature
```

This will:
1. Generate baseline dataset (5k transactions)
2. Train and register model to MLflow
3. Promote model to Production stage
4. Generate drifted dataset (feature drift)
5. Detect drift using Evidently
6. Execute control plane response
7. Log all decisions to `shared/events/`

---

## Dashboard

`make up` opens the dashboard automatically. It can also be opened standalone with `make dashboard`.

**What you can do from the dashboard:**

- **Current model card** — always-visible status strip showing the loaded model name, version, and stage. Updates instantly after a reload.
- **Try the API** — send a prediction request with custom transaction features and see the fraud probability live.
- **Full demos** — one-click end-to-end runs (`demo-drift-feature`, `demo-black-friday`). Each demo generates data → trains → promotes → monitors → triggers the control plane → reloads the API automatically.
- **Pipeline steps** — run each stage individually in order:

```
[1 Gen Reference] → [2 Gen Current] → [3 Train] → [4 Promote]
[5 Monitor]       → [6 Control]     → [7 Reload]
```

All commands stream live output in a side panel. The model status card updates after every reload so you can see the version change in real time.

---

## Demo Scenarios

### Feature Drift
Transaction amounts and distances shift significantly — the model learns different scale relationships.
```bash
make demo-drift-feature
```

### Shock Event (Black Friday)
Sudden late-night spike in transaction volume with elevated fraud rates. The retrained model weights night hours and high amounts more heavily.
```bash
make demo-black-friday
```

To see predictions change: after either demo completes, try the predict form with `Hour = 22` and `Amount = 1500`.

---

## How It Works

### 1. Synthetic Data Generation
Deterministic fraud dataset with configurable drift:
- **Features**: transaction_amount, customer_age, merchant_risk, geo_distance, etc.
- **Drift Types**: Feature distribution shifts, concept changes, shock events
- **Config-driven**: YAML files control drift parameters

### 2. Drift Detection
Evidently analyzes reference vs current data:
- Calculates drift ratio (% of features drifted)
- Classifies severity: none/low/medium/high
- Generates HTML report + JSON summary

### 3. Control Plane Execution

**Sentinel** reads monitoring summary:
```json
{
  "incident_type": "drift",
  "severity": "medium",
  "recommended_action": "retrain_and_evaluate"
}
```

**Planner** checks policies:
```yaml
drift_policy:
  on_medium:
    action: retrain_and_evaluate
```

**Release Agent** enforces quality gates:
```yaml
quality_gates:
  min_auc: 0.82
  min_average_precision: 0.20
```

If gates pass → promote to Staging/Production
If gates fail → block deployment, log reason

---

## Project Structure

```
driftwatch/
├── data/
│   └── generator/          # Synthetic fraud data generator
│       ├── generate.py
│       └── config/         # Drift scenario configs
├── services/
│   ├── api/                # FastAPI inference service
│   ├── training/           # Model training pipeline
│   ├── monitoring/         # Evidently drift detection
│   └── control_plane/      # Multi-agent orchestration
│       ├── agents/
│       │   ├── sentinel.py
│       │   ├── planner.py
│       │   └── release.py
│       ├── policies/
│       │   └── promotion.yaml
│       └── runner.py
├── infra/
│   ├── dashboard/          # Web dashboard (server.py + embedded HTML)
│   ├── prometheus/
│   └── grafana/
├── shared/                 # Runtime artifacts (gitignored)
│   ├── data/               # Generated datasets
│   ├── reports/            # Drift reports
│   └── events/             # Agent decision logs
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Policy Configuration

All automated decisions are governed by `services/control_plane/policies/promotion.yaml`:

```yaml
quality_gates:
  min_auc: 0.82
  min_average_precision: 0.20

drift_policy:
  on_none:
    action: noop
  on_low:
    action: noop
  on_medium:
    action: retrain_and_evaluate
  on_high:
    action: retrain_and_evaluate

release_policy:
  promote_if_quality_gates_pass: true
  promote_stage: Production
```

---

## Available Commands

```bash
# Infrastructure
make up              # Start services + open dashboard
make down            # Stop all services
make dashboard       # Open dashboard without restarting services
make logs            # Tail all logs
make api-logs        # Tail API logs only

# Data Generation
make gen-base        # Generate reference dataset
make gen-feature     # Generate feature drift current dataset
make gen-blackfriday # Generate Black Friday shock current dataset

# ML Pipeline
make train           # Train model and register to MLflow
make promote-prod    # Promote latest model to Production
make monitor         # Run Evidently drift detection
make control         # Execute control plane (sentinel → planner → release)
make reload-api      # Hot-reload Production model in the API

# End-to-End Demos
make demo-drift-feature   # Full pipeline: feature drift scenario
make demo-black-friday    # Full pipeline: Black Friday shock scenario

# Development (CI/CD)
make setup-dev       # Install dev dependencies
make install-hooks   # Install pre-commit hooks
make format          # Format code
make lint            # Run linting
make test            # Run tests
make check           # Run all checks
make ci-local        # Simulate CI pipeline
```

---

## Development & CI/CD

This project uses automated quality checks and testing:

**Setup (one-time):**
```bash
make setup-dev      # Install pytest, black, ruff, mypy, etc.
make install-hooks  # Install git pre-commit hooks
```

**Before committing:**
```bash
make format  # Auto-format code with Black & isort
make lint    # Check with Ruff & mypy
make test    # Run tests with coverage
```

Pre-commit hooks run automatically on `git commit`. GitHub Actions run on every push:
- Code formatting validation
- Linting and type checking
- Test suite with coverage
- Docker image builds

---

## Example: Inspecting Results

After running a demo:

```bash
# View drift summary
cat shared/reports/monitoring_summary.json

# View Sentinel's incident report
cat shared/events/sentinel_report.json

# View Planner's execution plan
cat shared/events/execution_plan.json

# View Release decision
cat shared/events/release_result.json

# Open drift report in browser
open shared/reports/drift_report.html
```

---

## License

MIT

---

## Acknowledgments

Built to demonstrate production-ready MLOps patterns for fraud detection systems. Not intended for actual fraud detection use.
