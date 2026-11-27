# Model and release behavior

[Back to README](../README.md)

## Synthetic payments

The generator models online card-payment attempts with seven inputs. These are
demo assumptions, not empirically validated fraud rules.

| Input | Meaning |
|---|---|
| `transaction_amount` | Payment amount in dollars |
| `transaction_hour` | Local hour, 0–23 |
| `payment_attempts_1h` | Prior attempts from the same session/device, including declines |
| `account_tenure_days` | Days since account creation |
| `merchant_risk_score` | Synthetic reputation index, 0–1; not a probability |
| `geo_distance_km` | Distance between payment IP location and the customer's usual IP location |
| `is_international` | Whether the payment is international |

Attempt counts are generated summaries; the demo does not maintain a transaction
ledger or collect device information. The distance is not distance to a merchant.

The default `baseline_fraud_rate: 0.002` targets 0.2% expected fraud on an
independent calibration population of 50,000 rows with seed 2026. For a given
fraud-logic configuration, the intercept stays fixed across batch sizes, seeds,
and scenarios. Current batches are never recalibrated to hide drift. Realized
fraud rates vary with sampling and population changes.

The baseline includes a 0.2% mixture of jointly risky payments and partial
variants so combined-risk examples have training support. Labels remain
probabilistic. A low overall fraud rate does not prevent an individual payment
from having a high probability.

## Features and fitting

New models use the `smooth_v2` transforms in
[`services/common/features.py`](../services/common/features.py). Amount,
distance, and attempts use `x² × (3 − 2x)`, with `x` clamped to 0–1. Their effects
saturate at $300, 200 km, and 20 attempts. Values and slopes join smoothly at the
endpoints; there is no abrupt switch at $150 or ten attempts.

The repeated-small-payment interaction is strongest up to $5, fades between $5
and $30, and is zero above $30. Local hour follows a circular night curve: a peak
from 23:00 through 05:00, falling to zero at 10:00 and 18:00. Account-tenure risk
decays as `exp(-days / 180)`.

Combined risk multiplies merchant risk, tenure risk, and the distance curve.
Amount and night-time each contribute a smooth multiplier from 0.5 to 1;
international status contributes 0.5 or 1. The default generator weights this
interaction by `compound_risk_weight: 12.0` in its log-odds calculation.

Training fits L2 logistic regression with nonnegative feature coefficients and
an unrestricted intercept. Higher merchant risk, distance, attempts, and
international status cannot reduce the fitted risk; longer tenure cannot
increase it. Amount can decrease risk within the small-payment region because
of the card-testing interaction. Above that region, increasing amount cannot
reduce risk.

Each training batch is split into 60% model fitting, 20% threshold selection,
and 20% release evaluation. The candidate and incumbent are evaluated on that
same final holdout. MLflow records the threshold, evaluation metrics, compared
Production version, `model_type: positive_logreg`, and feature version.

## Scenarios

Configurations live in [`data/generator/config/`](../data/generator/config/).
All four supplied configurations use 200,000 rows, with seeds 42–45.

- **Feature drift:** amount scales by 1.6, distance by 1.8, and merchant risk
  increases by 0.1. The fraud rule stays fixed. This is a single changed batch,
  not a gradual time series.
- **Black Friday:** amounts scale by 2.2 during hours 20–23, and fraud probability
  in that window is multiplied by 1.35. Merchant risk increases by 0.12 across
  the batch; the international rate becomes 0.28. Hours remain uniformly
  distributed, so there is no within-window volume surge.
- **Card testing:** about 1% of rows across all hours have amounts scaled to 5%
  and clipped to $1–$10, with 10–30 prior attempts and 4 added to their fraud
  log-odds. Unselected rows keep their features and labels for the same seed.
  Ordinary small or overnight payments are not automatically attacks.

Card-testing defaults come from `SHOCK_DEFAULTS` in
[`generate.py`](../data/generator/generate.py). Override `affected_fraction`,
`spike_hours`, `amount_scale`, or `attack_log_odds` under `drift` in the YAML.
Setting `affected_fraction: 0` disables the default attack.

With supplied configs and dependencies, feature drift and Black Friday retain
the baseline; card testing promotes the candidate. The integration tests cover
these decisions. Inspect the actual metrics in MLflow and
`shared/events/release_result.json` when changing a configuration.

## Predictions and thresholds

The dashboard shows probability on an unscaled 0–100% bar. Its summary says
**Safe** below 50% and **Fraud** at or above 50%. Descriptions use six bands:
below 10%, 10–30%, 30–50%, 50–70%, 70–90%, and 90–100%; a boundary belongs to the
higher band. These are labels for synthetic model estimates, not confirmed
outcomes. Editing an input hides the previous prediction until the next request.

The API's `is_fraud` flag instead uses the model's validation-selected threshold,
which minimizes `false positives + 100 × false negatives`. The cost ratio is
illustrative, not monetary, and can flag payments below 50%. The response includes
`decision_threshold`; MLflow records precision, recall, and the flagged fraction
on the release holdout. Calibration against synthetic labels does not establish
real-world calibration.

## Monitoring and release

[`promotion.yaml`](../services/control_plane/policies/promotion.yaml) is the
source of truth for alert thresholds and release gates.

Monitoring compares all seven input features with Evidently, excluding the label
from drift calculations. Feature severity is `none` at zero drift, `low` below
20%, `medium` from 20% to below 50%, and `high` at 50% or above. Any critical-feature,
outcome, or model-quality alert raises `incident_severity` to at least medium.

The registered Production version is loaded once and used for both reference
and current quality metrics. These include precision, recall, ROC AUC, average
precision, Brier score, log loss, ten-bin calibration error, and reliability bins.
Metrics are overall only; hourly slices have too few positive examples.

Default monitoring requirements and alerts:

| Check | Default |
|---|---|
| Critical-feature drift | Amount, attempts, or merchant risk |
| Outcome support | At least 100 labelled rows per window |
| Model-quality support | Also at least 20 examples of each class per window |
| Precision / recall decrease | More than 0.10 |
| AUC decrease | More than 0.05 |
| Brier / calibration-error increase | More than 0.002 / 0.005 |
| Fraud-rate change | More than 0.002 in either direction |

Changes are absolute differences, not significance tests. Labels may be absent
or partially missing; only labelled rows contribute to quality metrics, and
coverage is reported. Undefined metrics are `null`. An unavailable model is
reported explicitly while feature and observed-outcome monitoring continue.
Invalid labels or model probabilities fail the job.

Sentinel reads the summary, and Planner applies the severity policy: no action
for none/low, retrain and evaluate for medium/high. Incomplete current labels or
fewer than five examples of either class produce `await_labels` instead.

Release evaluates the registered candidate from that training run. It requires:

- AUC of at least 0.65 and average precision of at least 0.01.
- At least 50 examples of each class in the release holdout.
- At least 2% relative Brier-score improvement over Production, with AUC and
  average precision each falling by no more than 0.01.

Missing comparison evidence or a changed Production version blocks promotion.
Without an incumbent, the absolute gates still apply. These are demo tolerances;
they are not statistical significance tests. Manual `make promote-prod` bypasses
these gates to bootstrap the baseline.

Monitoring's registered model can differ from the API's loaded version until a
reload. Reusing training data as the reference window makes its quality metrics
optimistic; partially observed labels can be biased too. Use independent labelled
windows for meaningful comparisons. Candidate release evaluation is separate.

## Older models

Named legacy transforms remain available because serialized MLflow models refer
to them. Reloading an old model preserves its original behavior; it does not
apply the continuous curves to old coefficients. Compatible rollbacks likewise
restore the old behavior.

Models and CSVs using `customer_age` are incompatible with the current
`payment_attempts_1h` schema. To update an older installation, rebuild the API and
run a full demo to regenerate data, train, promote, and load a fresh baseline:

```bash
docker compose up -d --build api
make demo-drift-feature
```

Do not roll back to an age-based model. The API rejects that schema.
