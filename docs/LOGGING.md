# Structured Logging

DriftWatch uses `structlog` for professional, structured logging across all services.

## Features

- **Consistent Format**: All services use the same logging configuration
- **Structured Data**: Logs include contextual key-value pairs for easy filtering
- **Color-Coded Output**: Different log levels have distinct colors for readability
- **Service Identification**: Every log includes the service name (api, training, sentinel, etc.)
- **Production-Ready**: Can switch to JSON logs for production log aggregation

## Log Levels

- **DEBUG**: Detailed diagnostic information (prediction details, intermediate steps)
- **INFO**: Normal operational messages (service starting, tasks completing)
- **WARNING**: Potential issues detected (drift found, quality gates failed)
- **ERROR**: Errors that need attention (model not found, training failed)

## Example Logs

### Data Generator
```
2026-01-05 15:30:12 [info     ] Starting data generation        n_rows=5000 seed=42 drift_type=feature service=data_generator
2026-01-05 15:30:12 [info     ] Applied feature drift           amount_scale=1.6 distance_scale=1.8 merchant_risk_shift=0.1 service=data_generator
2026-01-05 15:30:13 [info     ] Data generation complete        fraud_count=487 fraud_rate=9.7% total_transactions=5000 service=data_generator
```

### Training Service
```
2026-01-05 15:31:45 [info     ] Starting training pipeline      experiment=fraud-demo model_name=fraud_detector service=training
2026-01-05 15:31:45 [info     ] Loading training data           path=/app/shared/data/reference.csv service=training
2026-01-05 15:31:45 [info     ] Data loaded                     fraud_rate=9.8% rows=5000 service=training
2026-01-05 15:31:45 [info     ] Dataset split                   test_size=1000 train_size=4000 service=training
2026-01-05 15:31:46 [info     ] Model evaluation complete       auc=0.6463 average_precision=0.4410 service=training
2026-01-05 15:31:46 [info     ] Training pipeline complete      run_id=fa024b7013e540b38442eb8c3842306a service=training
```

### Monitoring Service
```
2026-01-05 15:32:10 [info     ] Starting drift detection        current=/app/shared/data/current.csv reference=/app/shared/data/reference.csv service=monitoring
2026-01-05 15:32:10 [info     ] Datasets loaded                 current_rows=5000 reference_rows=5000 service=monitoring
2026-01-05 15:32:12 [info     ] Drift detection complete        drift_ratio=37.5% drifted_features=3 severity=medium total_features=8 service=monitoring
```

### Control Plane Agents

**Sentinel:**
```
2026-01-05 15:32:15 [info     ] Sentinel agent starting analysis report_dir=/app/shared/reports service=control_plane
2026-01-05 15:32:15 [warning  ] Incident detected               drifted_features=3 drift_ratio=0.375 incident_type=drift recommended_action=retrain_and_evaluate severity=medium service=control_plane
```

**Planner:**
```
2026-01-05 15:32:15 [info     ] Planner agent analyzing sentinel report incident_type=drift severity=medium service=control_plane
2026-01-05 15:32:15 [info     ] Execution plan created          action=retrain_and_evaluate incident_type=drift severity=medium service=control_plane
```

**Release Agent:**
```
2026-01-05 15:32:20 [info     ] Release agent evaluating latest model experiment=fraud-demo model_name=fraud_detector service=control_plane
2026-01-05 15:32:20 [info     ] Latest model metrics retrieved  auc=0.6463 average_precision=0.4410 run_id=fa024b7013e540b38442eb8c3842306a service=control_plane
2026-01-05 15:32:20 [warning  ] Quality gates failed - promotion blocked ap_gap=0.0000 auc=0.6463 auc_gap=0.1737 average_precision=0.4410 min_average_precision=0.2000 min_auc=0.8200 service=control_plane
```

## Viewing Logs

### Local Development
```bash
# View all logs
make logs

# View specific service
make api-logs

# View control plane execution
docker compose --profile jobs run --rm control_plane python /app/services/control_plane/runner.py
```

### Production Mode

For production, enable JSON logs for log aggregation (ELK, Splunk, etc.):

```python
# In each service
configure_logging("service_name", json_logs=True)
```

JSON output:
```json
{
  "event": "Model evaluation complete",
  "auc": "0.6463",
  "average_precision": "0.4410",
  "service": "training",
  "timestamp": "2026-01-05T15:31:46.123456Z",
  "level": "info"
}
```

## Adding Logging to New Code

```python
from services.common.logging import get_logger

logger = get_logger(__name__)

# Simple message
logger.info("Task started")

# With context
logger.info("Processing transaction",
            transaction_id=123,
            amount=250.00,
            is_fraud=True)

# Warnings
logger.warning("Threshold exceeded",
               current_value=0.95,
               threshold=0.90)

# Errors with exception info
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e), exc_info=True)
```

## Best Practices

1. **Use Structured Data**: Always include relevant context as key-value pairs
   ```python
   # Good
   logger.info("Model trained", auc=0.85, training_time_seconds=42)

   # Avoid
   logger.info(f"Model trained with AUC {auc} in {time}s")
   ```

2. **Choose Appropriate Levels**:
   - INFO: Normal flow (started, completed, loaded)
   - WARNING: Unexpected but handled (drift detected, gates failed)
   - ERROR: Actual failures (file not found, network error)

3. **Don't Log Sensitive Data**: Never log PII, credentials, or tokens

4. **Format Numbers Consistently**:
   ```python
   logger.info("Metrics", auc=f"{auc:.4f}", fraud_rate=f"{rate:.1%}")
   ```

5. **Include Identifiers**: Always log IDs for traceability
   ```python
   logger.info("Training started", run_id=run.info.run_id, experiment=exp_name)
   ```
