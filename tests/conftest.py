from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies not available in the test environment
# (mlflow, evidently, prometheus_client).  These must be registered before
# any service module is imported so that module-level imports succeed.
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> MagicMock:
    """Register a MagicMock stub in sys.modules if not already present."""
    if name not in sys.modules:
        mod = MagicMock()
        mod.__name__ = name
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


for package, submodules in {
    "mlflow": ("tracking", "sklearn", "pyfunc"),
    "evidently": ("metric_preset", "report"),
    "prometheus_client": (),
}.items():
    if importlib.util.find_spec(package) is None:
        module = _stub_module(package)
        for submodule in submodules:
            setattr(module, submodule, _stub_module(f"{package}.{submodule}"))
        if package == "prometheus_client":
            module.generate_latest.return_value = b"# metrics\n"


@pytest.fixture()
def tmp_report_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports"


@pytest.fixture()
def monitoring_summary_factory(tmp_report_dir: Path):
    """Write a monitoring_summary.json with given severity and return the dir."""

    def _make(severity: str, drift_ratio: float = 0.3, drifted_features: int = 2) -> Path:
        tmp_report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "drift_ratio": drift_ratio,
            "drifted_features": drifted_features,
            "total_features_checked": 7,
            "severity": severity,
        }
        (tmp_report_dir / "monitoring_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return tmp_report_dir

    return _make


@pytest.fixture()
def promotion_policy_path() -> str:
    """Return absolute path to the real promotion policy YAML."""
    return str(
        Path(__file__).parent.parent / "services" / "control_plane" / "policies" / "promotion.yaml"
    )
