# Logging

[Back to README](../README.md)

Services use `structlog` through
[`services/common/logging.py`](../services/common/logging.py). Entrypoints
configure INFO-level output to stdout, with timestamps, logger names, and a
bound `service` field. Console output uses colors. DEBUG events, including
prediction details, are filtered out by default.

## Read logs

```bash
make logs       # Running Compose services
make api-logs   # Inference API only
make control   # Run a new control-plane cycle and stream its output
```

Batch jobs print to the invoking terminal. In the dashboard, expand **Technical
output** in the progress panel. To inspect decisions after a job exits, use the
JSON files listed in [Operations](OPERATIONS.md#services-and-stored-data).

Useful release events include `Quality gates failed - promotion blocked`,
`Candidate did not improve on Production`, and `Candidate improved on Production`.
Comparison events include metric differences and failed checks. A retained
baseline can be a successful policy decision, rather than a job failure.

The dashboard parses lifecycle event names to track progress. Keep them stable,
or update the parser and progress tests with any rename.

## Add a logger

Configure logging once at the service entrypoint, then obtain module loggers:

```python
from services.common.logging import configure_logging, get_logger

configure_logging("training")
logger = get_logger(__name__)
logger.info("Model evaluated", run_id=run_id, auc=auc, evaluation_rows=rows)
```

Keep the event message short and place context in fields. Use INFO for normal
progress, WARNING for handled problems or blocked releases, and ERROR for failed
operations. Include run/model/version identifiers when available. Avoid logging
credentials, tokens, or individual payment details. Keep numeric fields numeric
when downstream consumers need to aggregate them.

For JSON output, change the entrypoint call to
`configure_logging("training", json_logs=True)`. This is a Python configuration
option; there is no `JSON_LOGS` environment switch. It changes log rendering,
not the service's deployment or security properties.
