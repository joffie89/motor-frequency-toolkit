![tests](https://github.com/joffie89/motor-frequency-toolkit/actions/workflows/ci.yml/badge.svg)

# motor-frequency-toolkit

A reusable Python toolkit for **motor claim-frequency modelling**: from raw policy data to a
model comparison, a feature-selection report and segment-level monitoring, with a human
approval gate before any model is promoted.

> **Data and scope.** Uses the public French motor dataset *freMTPL2freq* from OpenML (678,013 policies).
> It covers all claim types and has no dates, so **time periods and the emerging-risk shift are
> simulated and illustrative**. The aim is to show the approach and the engineering, not real results.

## Why this exists

When a pricing team builds or rebuilds a frequency model, the same steps are often redone by hand
in notebooks: preparing data, checking rating factors, comparing models and checking performance
over time. This toolkit packages those steps so they are **consistent, configurable, tested and
reviewable**.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
python -m motorfreq run         # downloads data, runs everything, writes reports/
pytest -q                       # 10 tests
```

All settings are in `configs/default.yaml`.

## What it does

| Step | Module | Question it answers |
| --- | --- | --- |
| Load and clean | `data.py` | Claims per **vehicle-year**: frequency = claims ÷ exposure (insured time) |
| Quality checks | `validate.py` | Is the data safe to model? (missing values, exposure in (0, 1], claims ≥ 0) |
| Features | `features.py` | How should 22 regions be grouped so sparse areas aren't over-trusted? |
| Models | `models.py` | Poisson GLM (champion) vs Poisson GBM (challenger), weighted by exposure |
| Evaluation | `evaluate.py` | Fit (Poisson deviance), calibration (O/E) and risk separation (lift) |
| Feature selection | `feature_selection.py` | Which rating factors help, which overlap, which are unstable? |
| Monitoring | `monitor.py` | Which segments are drifting away from predictions? |
| Pipeline | `pipeline.py` | Runs it all; **recommends** a model but waits for a named approval |

## Results (illustrative)

From `python -m motorfreq run` with the default config (OpenML data; overall frequency 0.104 claims per vehicle-year).

**Champion vs challenger**, trained on periods 1–3, tested on period 4:

| Model | Poisson deviance | O/E | Lift (top ÷ bottom decile) |
| --- | --- | --- | --- |
| Poisson GLM | 0.6424 | 1.133 | 7.96× |
| Poisson GBM | 0.6281 | 1.122 | 9.55× |

The GBM fits better and separates risk more sharply. O/E is above 1 for both because the test period
contains the simulated risk increase, which no model trained on earlier data can know about.

![Lift chart](reports/lift_chart.png)

**Why time-based validation matters.** One vehicle segment (brand B12) gets a simulated +60%
frequency rise in the latest period, standing in for something like a new theft method:

| Validation | Segment O/E |
| --- | --- |
| Train on earlier periods, test on the latest | **1.56**: the shift is visible |
| Random 75/25 split | **1.05**: the shift is hidden |

A random split mixes old and new data, so it can make a stale model look fine.

**Location grouping.** Credibility-weighted grouping of 22 regions into 5 bands matches the fit of the
raw regions (deviance 0.6424 vs 0.6421) with slightly better lift (7.96× vs 7.91×), and improves lift
over leaving region out (7.79×).

**Feature selection** (`reports/feature_selection.csv`) combines three views per feature:
how much validation deviance worsens without it, how strongly frequency varies across its values,
and how stable its effect is over time. It flags **VehBrand as unstable** (the feature affected by the
simulated shift) even though it still helps the model, and shows that **Area, Density and the region
groups** each add little on their own because they carry overlapping location information.

**Monitoring** (`reports/monitoring.csv`) raises an alert for B12 (O/E 1.56 against a 1.2 threshold).

## Approval gate

The pipeline writes `reports/decision.json` with `"status": "awaiting_approval"`. Nothing is promoted
automatically. A reviewer records approval explicitly:

```bash
python -m motorfreq approve --by "Reviewer Name"
```

## Engineering

- Installable package (`pyproject.toml`), configuration in YAML, no hard-coded settings
- 10 `pytest` tests on synthetic data (fast, no download), run on every push by GitHub Actions
- Training-only fitting of groupings to avoid leakage into the test period

## Next steps

- Real policy dates and claim development (reporting delays) instead of simulated periods
- SHAP explanations for the GBM, and fairness checks for location features
- Severity modelling alongside frequency
