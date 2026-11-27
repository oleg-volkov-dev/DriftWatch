# Development

[Back to README](../README.md)

## Set up

Use Python 3.11, matching the containers and CI. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup-dev
make install-hooks
```

`setup-dev` installs both requirements files. `install-hooks` is optional and
installs the repository's pre-commit checks. Node.js is needed for the dashboard
JavaScript tests; those tests skip when Node is unavailable. Docker is needed
for Grafana query checks and container builds.

## Check a change

```bash
make format
make lint
make test
make test-grafana
```

| Target | What it does |
|---|---|
| `make format` | Apply Ruff lint fixes, including import sorting; it does not run the Ruff formatter |
| `make lint` | Run Ruff and mypy on the configured paths |
| `make test` | Run pytest with terminal and HTML coverage reports |
| `make test-grafana` | Run Prometheus query fixtures in Docker |
| `make check` | Run all four targets above; this can edit files through `format` |
| `make ci-local` | Run import/lint checks, tests, and four Docker builds; omits mypy and Grafana checks |

For layout changes to Python code, run `ruff format` on the files you edited.
The configured line length is 100. Keep related arguments on separate lines when
it makes validation rules or model transforms easier to follow. Comments should
explain constraints and decisions; avoid restating the next line of code.

Run a focused test file with `pytest tests/test_api.py`. `make test` writes an
HTML coverage report to `htmlcov/index.html`.

## Test coverage and limits

The suite covers feature curves, generator behavior, model fitting, API reloads,
monitoring, planner decisions, release gates, rollback, and dashboard progress.
The real-dependency lifecycle tests use temporary MLflow storage and exercise
all three default scenario decisions, so they do not need the running demo stack.

When MLflow, Evidently, or Prometheus packages are unavailable, `tests/conftest.py`
provides import stubs for unit tests. The lifecycle tests skip when MLflow is
missing. Install both requirements files for a complete run; a passing partial
suite is not evidence that the real service integrations work.

Dashboard logic tests execute extracted JavaScript with Node. They do not test
browser layout. After changing HTML or CSS, open the local dashboard and inspect
it at desktop and narrow widths.

## CI and hooks

[CI](../.github/workflows/ci.yml) runs on pushes and pull requests to `main` and
`develop`. It checks Ruff linting, pytest coverage, Grafana query fixtures, four
Docker builds, and a Compose smoke test that generates data, trains a model,
and verifies registry creation. It does not run mypy or `ruff format --check`.

[The publish workflow](../.github/workflows/docker-publish.yml) builds service
images for Linux amd64 and arm64. It publishes to GHCR on `main` pushes and
version tags; pull requests only build.

[Pre-commit](../.pre-commit-config.yaml) checks whitespace, structured files,
merge markers, private keys, Ruff fixes, and mypy. Hooks may edit files, so review
the diff before committing.

## Repository conventions

Keep reusable documentation in `README.md` or `docs/`, tests in `tests/`, and
personal notes or scratch output in `.local/`. Generated data, models, reports,
coverage, caches, `.env` files, and assistant state are ignored by Git and
excluded from Docker build contexts.

The Python feature functions named in serialized models are a compatibility
boundary. Preserve legacy transforms when introducing new ones; changing their
meaning changes predictions after a reload or rollback. Behavior tests should
cover changes to those curves and thresholds.

The dashboard reads lifecycle log messages to show progress. If you change an
event name, update the progress parser and its tests together. See
[Logging](LOGGING.md) for the shared logger configuration.
