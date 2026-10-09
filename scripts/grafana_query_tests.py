"""Emit promtool test cases using the actual provisioned dashboard queries.

JSON is also valid YAML, so the output can be passed straight to promtool.
Run with `make test-grafana` (requires Docker).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = json.loads(
    (ROOT / "infra/grafana/provisioning/dashboards/fraud_api.json").read_text()
)
PANELS = {panel["id"]: panel for panel in DASHBOARD["panels"]}


def query(panel_id, target=0):
    return (
        PANELS[panel_id]["targets"][target]["expr"]
        .replace("$__range", "5m")
        .replace("$__rate_interval", "1m")
    )


def check(panel_id, value=None, labels="{}", target=0):
    return {
        "expr": query(panel_id, target),
        "eval_time": "5m",
        "exp_samples": [] if value is None else [{"labels": labels, "value": value}],
    }


def series(metric, values, labels='job="api",instance="api:8000"'):
    return {"series": f"{metric}{{{labels}}}", "values": values}


def cases():
    # Every expression must parse, and missing metrics must not become healthy zeros.
    tests = [{
        "name": "No exporters or monitoring results",
        "interval": "15s",
        "input_series": [],
        "promql_expr_test": [
            check(panel["id"], target=i)
            for panel in DASHBOARD["panels"]
            for i, _ in enumerate(panel.get("targets", []))
        ],
    }]
    counters = [
        series("api_requests_total", "0+0x20"),
        series("api_errors_total", "0+0x20"),
        series("api_request_latency_seconds_sum", "0+0x20"),
        series("api_request_latency_seconds_count", "0+0x20"),
        series("fraud_predictions_total", "0+0x20", 'job="api",result="fraud"'),
        series("fraud_predictions_total", "0+0x20", 'job="api",result="legit"'),
    ]
    counters += [
        series(metric, "0+0x20", f'job="api",le="{bound}"')
        for metric in ("fraud_score_bucket", "api_request_latency_seconds_bucket")
        for bound in ("1", "+Inf")
    ]
    tests.append({
        "name": "Idle API: zero counts, undefined fractions and mean latency",
        "interval": "15s",
        "input_series": counters,
        "promql_expr_test": [
            check(1, 0), check(2, 0), check(3), check(4), check(6),
            check(8), check(9), check(10), check(12),
        ],
    })
    tests.append({
        "name": "Counter reset within selected range is counted correctly",
        "interval": "15s",
        "input_series": [series("api_requests_total", "0+1x10 0+1x9")],
        "promql_expr_test": [check(1, 19)],
    })
    tests.append({
        "name": "Multiple API instances aggregate before fractions and quantiles",
        "interval": "15s",
        "input_series": [
            series(metric, values, f'job="api",instance="{instance}"{extra}')
            for instance in ("a", "b")
            for metric, values, extra in [
                ("api_requests_total", "0+10x20", ""),
                ("api_errors_total", "0+1x20", ""),
                ("api_request_latency_seconds_sum", "0+1x20", ""),
                ("api_request_latency_seconds_count", "0+10x20", ""),
                ("fraud_predictions_total", "0+3x20", ',result="fraud"'),
                ("fraud_predictions_total", "0+6x20", ',result="legit"'),
                ("fraud_score_bucket", "0+0x20", ',le="0"'),
                ("fraud_score_bucket", "0+9x20", ',le="1"'),
                ("fraud_score_bucket", "0+9x20", ',le="+Inf"'),
            ]
        ],
        "promql_expr_test": [
            check(1, 400), check(2, 40), check(3, .1), check(4, .1),
            check(8, 1 / 3), check(9, .5), check(10, .95),
        ],
    })
    tests.append({
        "name": "Only unflagged predictions produce a zero flagged fraction",
        "interval": "15s",
        "input_series": [
            series("fraud_predictions_total", "0+0x20", 'job="api",result="fraud"'),
            series("fraud_predictions_total", "0+1x20", 'job="api",result="legit"'),
        ],
        "promql_expr_test": [check(8, 0)],
    })
    tests.append({
        "name": "Down API does not show a cached ready model",
        "interval": "15s",
        "input_series": [series("up", "0+0x20"), series("api_model_loaded", "1+0x20")],
        "promql_expr_test": [check(18, 0), check(19)],
    })
    tests.append({
        "name": "Card-testing batch and age of last successful push",
        "interval": "15s",
        "input_series": [
            series(metric, values, 'job="driftwatch_monitoring"')
            for metric, values in [
                ("monitoring_drift_ratio", f"{1 / 7}+0x20"),
                ("monitoring_drift_severity", "1+0x20"),
                ("monitoring_drifted_features", "1+0x20"),
                ("monitoring_total_features", "7+0x20"),
                ("push_time_seconds", "120+0x20"),
            ]
        ],
        "promql_expr_test": [
            check(14, 1 / 7, 'monitoring_drift_ratio{job="driftwatch_monitoring"}'),
            check(15, 1, 'monitoring_drift_severity{job="driftwatch_monitoring"}'),
            check(16, 1, 'monitoring_drifted_features{job="driftwatch_monitoring"}'),
            check(16, 7, 'monitoring_total_features{job="driftwatch_monitoring"}', target=1),
            check(21, 180),
        ],
    })
    return {"evaluation_interval": "15s", "tests": tests}


if __name__ == "__main__":
    print(json.dumps(cases()))
