# DriftWatch

**Autonomous MLOps platform for fraud detection with policy-driven drift response**

A production-grade MLOps demonstration that automatically detects, evaluates, and responds to model drift using a multi-agent control plane. Built with synthetic fraud data to showcase real-world ML reliability patterns.

---

## Overview

DriftWatch simulates a complete ML lifecycle where models monitor themselves and autonomously decide when to retrain, deploy, or rollback based on drift detection and quality gates.

### Key Features

- **Automated Drift Detection**: Monitors feature and concept drift using Evidently
- **Multi-Agent Control Plane**: Three specialized agents coordinate automated responses
  - **Sentinel**: Observes metrics and classifies incidents
  - **Planner**: Generates execution plans based on policies
  - **Release**: Enforces quality gates and safe deployments
- **Policy-as-Code**: All decisions constrained by versioned YAML policies
- **Full Observability**: MLflow tracking, Prometheus metrics, drift reports
- **Deterministic Demos**: Reproducible drift scenarios for testing

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

Access UIs:
- **MLflow**: http://localhost:5000
- **API**: http://localhost:8000/docs
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

## Demo Scenarios

### Feature Drift
Transaction amounts and distances shift significantly.
```bash
make demo-drift-feature
```

### Concept Drift
Fraud patterns change (e.g., fraud moves from international to nighttime transactions).
```bash
make demo-drift-concept
```

### Shock Event (Black Friday)
Sudden spike in transaction volume and amounts.
```bash
make demo-black-friday
```

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
  promote_stage: Staging
```

---

## Available Commands

```bash
# Infrastructure
make up              # Start core services
make down            # Stop all services
make logs            # Tail all logs
make api-logs        # Tail API logs only

# Data Generation
make gen-base        # Generate reference dataset
make gen-feature     # Generate feature drift
make gen-concept     # Generate concept drift
make gen-blackfriday # Generate shock event

# ML Pipeline
make train           # Train model on reference data
make promote-prod    # Promote latest to Production
make monitor         # Run drift detection
make control         # Execute control plane

# End-to-End Demos
make demo-drift-feature
make demo-drift-concept
make demo-black-friday
```

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

## What This Demonstrates

### Technical Skills
- End-to-end ML pipeline orchestration
- Production monitoring and alerting
- Automated model retraining workflows
- Quality gates and release strategies
- Policy-driven decision making

### MLOps Maturity
- **Level 0**: Manual training and deployment ❌
- **Level 1**: Automated training pipeline ✅
- **Level 2**: Automated CI/CD for ML ✅
- **Level 3**: Automated retraining triggers ✅
- **Level 4**: Automated model monitoring & response ✅ (This project)

### Best Practices
- Deterministic, reproducible experiments
- Model versioning and registry
- Drift detection and alerting
- Automated testing and validation
- Auditability and observability
- Safe deployment strategies

---

## Future Enhancements

- [ ] Performance degradation detection (precision/recall drop)
- [ ] Canary deployment simulation (traffic split)
- [ ] A/B testing framework
- [ ] GitHub PR-based approval flow
- [ ] OpenTelemetry distributed tracing
- [ ] Kubernetes deployment with Helm
- [ ] Real-time streaming inference
- [ ] Model explainability reports

---

## License

MIT

---

## Acknowledgments

Built to demonstrate production-ready MLOps patterns for fraud detection systems. Not intended for actual fraud detection use.
