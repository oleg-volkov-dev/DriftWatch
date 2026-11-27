# Operations

[Back to README](../README.md)

Run commands from the repository root. The dashboard invokes the same Make
commands and permits one pipeline job at a time. Avoid running overlapping
terminal jobs: datasets and decision files use shared paths.

## Pipeline commands

| Command | Effect |
|---|---|
| `make gen-base` | Write baseline data to `shared/data/reference.csv` |
| `make gen-feature` | Write feature-drift data to `shared/data/current.csv` |
| `make gen-blackfriday` | Write Black Friday data to `shared/data/current.csv` |
| `make gen-card-testing` | Write card-testing data to `shared/data/current.csv` |
| `make train` | Train and register a model from reference data |
| `make promote-prod` | Manually promote the latest registered version, bypassing release gates |
| `make monitor` | Compare reference/current data and Production model quality |
| `make control` | Read the monitoring report, plan a response, and evaluate a candidate if requested |
| `make reload-api` | Load the registered Production model into the API |
| `make rollback` | Restore the version in the rollback manifest, then reload the API |

To try card testing against the existing Production model:

```bash
make gen-card-testing
make monitor
make control
make reload-api
```

This assumes reference data and a Production model already exist. For a first
run, use a full demo from the [quick start](../README.md#quick-start).

The control plane trains on current data when it is available. A rejected
candidate leaves Production unchanged. Expected policy rejections can finish
successfully; inspect the decision, not just the command's exit code.

A control-plane promotion that replaces Production writes a rollback manifest.
Rollback checks both the model name and expected Production version, restores
the previous version, and consumes the manifest. Manual baseline promotion does
not create this manifest.

## Services and stored data

| Command | Effect |
|---|---|
| `make up` | Build/start core services and open the local dashboard |
| `make dashboard` | Start/restart the dashboard and open it; uses macOS `open` |
| `make logs` / `make api-logs` | Follow container logs |
| `docker compose stop` | Stop containers and retain their data; the host dashboard keeps running |
| `make down` | Remove containers and volumes, including MLflow history; stop the Make-started dashboard |
| `make build` | Rebuild all job and API images without the build cache |
| `make clean-shared` | Delete files under `shared/data`, `shared/reports`, and `shared/events` |

MLflow stores its database and model artifacts in the `mlflow_data` Docker volume.
Generated CSVs, reports, and control-plane decisions live under `shared/` on the
host. `make down` and `make clean-shared` therefore erase different data.

| File | What to inspect |
|---|---|
| `shared/reports/drift_report.html` | Feature-distribution comparisons |
| `shared/reports/monitoring_summary.json` | Drift, label coverage, quality metrics, alerts, and evaluated model |
| `shared/events/sentinel_report.json` | Classified incident and evidence |
| `shared/events/execution_plan.json` | Selected action and policy |
| `shared/events/training_result.json` | Candidate training run or error |
| `shared/events/release_result.json` | Promotion decision and comparison details |
| `shared/events/rollback_manifest.json` | Expected Production and rollback target versions |
| `shared/events/rollback_result.json` | Last rollback result |

These files are overwritten when their phase runs. A no-op cycle may leave older
training/release files in place; check the execution plan and run IDs before
interpreting them as the latest outcome. The HTML report covers feature drift;
model-quality details are in the JSON summary.

## Inference API

After training, promotion, and reload:

```bash
curl -sS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_amount": 20,
    "transaction_hour": 14,
    "payment_attempts_1h": 1,
    "account_tenure_days": 365,
    "merchant_risk_score": 0.2,
    "geo_distance_km": 10,
    "is_international": false
  }'
```

The response includes `fraud_probability`, `is_fraud`, `decision_threshold`, and
`model_stage`. The flag uses the model's saved review threshold, which can differ
from the dashboard's 50% summary. See [model behavior](MODEL.md#predictions-and-thresholds).

`GET /health` reports the loaded version and `model_loaded`. A 200 response alone
means the API is reachable, not that a model is ready. `POST /reload` resolves
Production and loads that version; a failed reload returns 503 while preserving
an already loaded model. `/predict` returns 503 without a model, 422 for invalid
input, and 500 if inference fails. Full input constraints are available at
[the API docs](http://localhost:8000/docs).

## Configuration

Compose reads `.env`, and Make includes it as well. Use simple `KEY=value` entries
compatible with both. The variables passed to the relevant containers are:

| Variable | Compose default |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` |
| `MLFLOW_EXPERIMENT_NAME` | `fraud-demo` |
| `MODEL_NAME` | `fraud_detector` |

When running Python services directly on the host, the default tracking URI is
`http://localhost:5000`. The Compose hostname `mlflow` is for container traffic.
Keep the model name consistent across training, monitoring, control, and API.

Edit [scenario YAML](../data/generator/config/) to change generated data and
[the policy](../services/control_plane/policies/promotion.yaml) to change alerts
or release gates. Job commands rebuild their images, so these edits are picked
up on the next run. API source changes need `docker compose up -d --build api`.

Direct control-plane runs also accept `POLICY_PATH`, `DATA_DIR`, `REPORT_DIR`, and
`EVENTS_DIR`. Compose fixes the shared paths and does not forward every host
variable. For a one-off container override, pass `docker compose run -e` and use
a path available inside the container. Monitoring accepts `--policy`,
`--report-dir`, and `--pushgateway`; see its `--help` for arguments.

## Grafana and metrics

Open [DriftWatch — API & Drift](http://localhost:3000/d/driftwatch-fraud-api).

- **Online activity** comes from `/predict` calls. Demos and data generation do
  not send prediction traffic. Attempts include inference errors and missing-model
  responses, but exclude input-validation errors (422).
- **Counts, error fraction, and score statistics** use the selected time range.
  Prometheus estimates counts across scrapes and counter resets, so they may be
  fractional. Empty fractions or percentiles mean no observations or too few
  scrapes. Graphs use an adaptive rolling window.
- **Flagged predictions** are scores above the loaded model's review threshold,
  not confirmed fraud. Score percentiles estimate a histogram, not accuracy.
  Execution latency excludes validation, queuing, and network time.
- **Offline monitoring** shows the latest successfully pushed batch at the
  selected end time. Values persist between runs. Check result age and gateway
  health before interpreting a quiet dashboard as healthy.
- **Health** distinguishes API reachability, a loaded model, and gateway
  reachability. Missing metrics are not filled in as healthy zeros.

Monitoring exports drift ratio/severity, per-feature drift, combined incident
severity, alert counts, and per-alert series. `monitoring_quality` uses
`window`, `scope`, and `metric` labels; scope is `overall`. Availability, label
coverage, and labelled row counts have their own series. Undefined metrics are
omitted, and each successful push replaces the previous batch.

Grafana dashboard files refresh automatically. After changing the API exporter,
rebuild the API; after data-source provisioning changes, restart Grafana. Run
`make monitor` to refresh batch metrics. `make test-grafana` checks the dashboard's
actual PromQL queries against fixtures.

## Troubleshooting

| Symptom | Next step |
|---|---|
| No model loaded | On a fresh install, run a full demo. Otherwise check Production in MLflow and run `make reload-api`. |
| MLflow connection fails after startup | Check `docker compose logs mlflow` and `curl -fsS http://localhost:5000/health`, then retry the job. |
| Candidate rejected | Read `release_result.json`; drift alone does not justify a replacement. |
| `await_labels` | Supply complete current labels with at least five rows of each class, then rerun monitoring and control. Release requires a larger holdout. |
| Rollback manifest is missing or stale | Check the current registry version and latest release decision. A rollback only applies to the promotion recorded in its manifest. |
| API version differs from Production | Run `make reload-api`; monitoring reads the registry independently. |
| Grafana shows old results | Check result age and Pushgateway health, then run `make monitor`. A failed push can leave old metrics even when local reports were saved. |
| Dashboard loses its connection | The job may still be running. Inspect its logs before starting another run. |
| Old `customer_age` schema | Follow the [migration steps](MODEL.md#older-models). |
