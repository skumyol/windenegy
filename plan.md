# Windenegy Development Plan

## 1. Product Thesis

Build an industry-grade wind power forecasting prototype that shows how meteorological intelligence can improve renewable energy asset decisions.

This should feel aligned with Stellerus: climate technology, satellite and weather data fusion, physics-guided AI, hazard-aware decision support, and operationally useful forecasts. The first version should not overclaim satellite foundation modeling or diffusion modeling. It should demonstrate the same philosophy at a smaller scale: combine trusted physical signals, rigorous evaluation, and deployable AI to turn climate data into decisions.

## 2. Target Outcome

The project should answer one concrete operational question:

> Given recent turbine SCADA observations and weather context, what power output should an operator expect over the next forecast horizon, and how uncertain is that forecast?

The demo should include:

- A reproducible data pipeline from raw SCADA/weather data to validated model-ready datasets.
- Strong baselines before deep learning.
- A sequence model for multi-horizon forecasting.
- Forecast uncertainty or prediction intervals.
- A FastAPI service with explicit request/response schemas.
- A dashboard focused on operator decisions, not model vanity metrics.
- Tests, linting, type checks, CI, Docker, and clear documentation.

## 3. Scope

### In Scope

- Single-turbine forecasting using the Kaggle Turkey wind turbine SCADA dataset.
- Optional weather enrichment using ERA5, Open-Meteo archive/forecast data, or another reproducible public weather source.
- Forecast horizons: 1 hour, 6 hours, and 24 hours.
- Models:
  - Persistence baseline.
  - Manufacturer/theoretical power curve baseline where available.
  - Gradient boosting baseline.
  - PatchTST or another maintained sequence model.
- Evaluation by horizon, wind regime, ramp events, and curtailment/anomaly periods.
- API and dashboard deployment-ready locally through Docker Compose.

### Out of Scope for V1

- Production satellite data ingestion.
- Real-time turbine control.
- Multi-farm dispatch optimization.
- Claims of operational accuracy beyond the public dataset.
- Custom foundation model training.

These can be positioned as future work once the core system is credible.

## 4. Stellerus Alignment

The project should map to Stellerus themes explicitly:

- Climate intelligence: forecasts are framed as decision support for climate-exposed assets.
- Data fusion: SCADA plus weather covariates, with a clean interface for future satellite/radar features.
- Physics-guided AI: encode known wind-power behavior through power-curve baselines, feature constraints, physical sanity checks, and regime-based evaluation.
- Risk and resilience: highlight ramp events, low-confidence windows, and potential energy shortfall risk.
- Practical platform thinking: API, dashboard, monitoring hooks, reproducible model registry, and clean docs.

Recommended README language:

> This prototype demonstrates weather-aware renewable energy forecasting: a small-scale analogue of climate intelligence systems that fuse observed asset data with meteorological context to support infrastructure decisions.

Avoid:

- Claiming remote-sensing expertise unless satellite/weather data is actually integrated.
- Claiming production readiness without CI, tests, versioned artifacts, and deployment instructions.
- Claiming PatchTST is automatically better before measured results exist.

## 5. System Architecture

Use a modular service layout with clear contracts:

```text
windenegy/
  pyproject.toml
  README.md
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
  configs/
    default.yaml
  data/
    raw/
    interim/
    processed/
  artifacts/
    models/
    metrics/
    reports/
  notebooks/
    01_data_audit.ipynb
    02_model_experiments.ipynb
  src/windenegy/
    __init__.py
    config.py
    schemas.py
    data/
      ingest.py
      validate.py
      split.py
      weather.py
    features/
      scada.py
      weather.py
      sequences.py
    models/
      baseline.py
      boosting.py
      patchtst.py
      registry.py
    evaluation/
      metrics.py
      backtest.py
      reports.py
    service/
      app.py
      dependencies.py
    dashboard/
      app.py
  tests/
    unit/
    integration/
    fixtures/
```

Architecture rules:

- Keep notebooks exploratory only. Production logic lives under `src/windenegy`.
- Use typed config objects instead of hard-coded paths and horizons.
- Treat feature generation as a versioned contract shared by training and inference.
- Treat trained models as artifacts with metadata: model type, feature schema, target, horizon, training window, metrics, git SHA.
- Do not let the API manually rebuild ad hoc feature arrays. It must call the same feature pipeline used during training.

## 6. Data Design

### Raw Data Contract

For the Kaggle SCADA dataset, expect:

- `Date/Time`
- `LV ActivePower (kW)`
- `Wind Speed (m/s)`
- `Theoretical_Power_Curve (KWh)`
- `Wind Direction (°)`

Normalize to internal names at ingestion:

- `timestamp`
- `active_power_kw`
- `wind_speed_mps`
- `theoretical_power_kwh`
- `wind_direction_deg`

### Validation Checks

Implement validation before modeling:

- Required columns exist.
- Timestamps parse and are monotonic after sorting.
- Duplicate timestamps are handled deterministically.
- Power is non-negative or flagged.
- Wind speed is physically plausible.
- Wind direction is in `[0, 360]`, modulo-normalized if needed.
- Missingness is reported by column and time period.
- Resampling behavior is explicit: keep 10-minute data for sequence models; optionally aggregate for baselines.

### Weather Enrichment

V1 can run without external weather data, but the architecture should support weather covariates. Add a provider interface:

- `WeatherProvider.fetch(start, end, latitude, longitude) -> WeatherFrame`
- `OpenMeteoWeatherProvider` as the first implementation if network/data access is available.
- `NullWeatherProvider` for deterministic local tests.

Potential weather features:

- 10m and 100m wind speed.
- Wind direction.
- Gusts.
- Temperature.
- Pressure.
- Boundary-layer or stability proxy if available.

## 7. Modeling Strategy

### Baselines First

Start with baselines because they anchor credibility:

- Persistence: future power equals most recent observed power.
- Rolling mean: recent average power.
- Theoretical power curve: compare observed power against provided curve.
- Gradient boosting: tabular model with lags, rolling windows, wind direction sin/cos, time features, and weather covariates.

No deep learning model should be accepted unless it beats relevant baselines on held-out chronological data.

### Sequence Model

Use a maintained PatchTST path if possible:

- Prefer Hugging Face `transformers` PatchTST for dependency stability.
- Use the original PatchTST repo only if a specific capability is unavailable elsewhere.

Sequence contract:

- Input window: recent 24 to 72 hours of 10-minute observations.
- Prediction length: 6, 36, and 144 steps for 1h, 6h, and 24h at 10-minute resolution.
- Target: `active_power_kw`.
- Known future covariates: time features and weather forecast features if available.
- Observed covariates: SCADA variables available only up to forecast creation time.

### Uncertainty

Include uncertainty in V1 if feasible:

- Quantile regression for boosting baselines, or
- Conformal prediction intervals over validation residuals, or
- Multiple quantile heads if the sequence model supports it.

Acceptance target:

- Report P50 and P90 forecasts.
- Measure interval coverage and average interval width.

## 8. Evaluation Plan

Use chronological splits only:

- Train: earliest 70 percent.
- Validation: next 15 percent.
- Test: final 15 percent.

Report:

- MAE, RMSE, MAPE or sMAPE by horizon.
- Skill score versus persistence.
- Error by wind-speed bin.
- Error during ramp events.
- Error during high-power and low-power regimes.
- Prediction interval coverage if uncertainty is implemented.

Visuals:

- Power curve: observed vs theoretical.
- Forecast overlays for representative weeks.
- Horizon error chart.
- Ramp-event case study.
- Calibration plot for intervals.

Definition of done for a model:

- Reproducible training command.
- Saved artifact with metadata.
- Test-set metrics written to `artifacts/metrics`.
- Feature schema recorded.
- API can load and serve the artifact.

## 9. API Design

FastAPI endpoints:

- `GET /health`
- `GET /metadata`
- `POST /forecast`

Forecast request should accept a sequence, not a single slider value:

```json
{
  "asset_id": "T1",
  "created_at": "2018-12-31T00:00:00Z",
  "horizon_hours": 6,
  "observations": [
    {
      "timestamp": "2018-12-30T23:50:00Z",
      "active_power_kw": 812.4,
      "wind_speed_mps": 7.2,
      "wind_direction_deg": 184.0,
      "theoretical_power_kwh": 900.1
    }
  ],
  "weather_forecast": []
}
```

Forecast response:

```json
{
  "asset_id": "T1",
  "model_version": "patchtst-2026-04-27",
  "horizon_hours": 6,
  "unit": "kW",
  "forecast": [
    {
      "timestamp": "2018-12-31T00:00:00Z",
      "p50": 820.2,
      "p10": 690.5,
      "p90": 960.8
    }
  ],
  "warnings": []
}
```

API quality requirements:

- Pydantic schemas for all external contracts.
- Clear 4xx errors for bad input.
- Model-load failure should fail startup, not first request.
- `/metadata` exposes feature schema and metric summary.
- Add request examples to OpenAPI docs.

## 10. Dashboard Design

Build the dashboard for an energy/climate operations audience:

- Forecast view: observed power, forecast P50, uncertainty band.
- Risk view: likely underproduction windows and ramp alerts.
- Model performance view: horizon metrics and baseline comparison.
- Data quality view: missing data, anomalies, curtailment-like periods.

Avoid a marketing hero page. The first screen should be the operational forecast.

## 11. Code Quality Standards

Use:

- `ruff` for linting and formatting.
- `mypy` or `pyright` for type checks.
- `pytest` for unit and integration tests.
- `pre-commit` for local checks.
- `pydantic` for config and API schemas.
- `polars` or `pandas`; pick one primary dataframe library and keep usage consistent.
- `scikit-learn` pipelines where possible for tabular models.

Rules:

- No model code in notebooks only.
- No hard-coded absolute paths.
- No hidden global mutable config.
- No training/inference feature drift.
- No unpinned core dependencies in final deliverable.
- No test that requires Kaggle credentials.
- No API endpoint that depends on a notebook artifact without validation metadata.

## 12. Test Strategy

Unit tests:

- Column normalization.
- Timestamp parsing and sorting.
- Wind direction sin/cos transform.
- Lag and rolling feature creation.
- Sequence window generation.
- Metric calculations.
- Pydantic request validation.

Integration tests:

- Raw fixture CSV to processed dataset.
- Train a tiny baseline model on fixture data.
- Save and reload model artifact.
- API `/forecast` returns expected schema.
- Dashboard import smoke test.

Data tests:

- Required schema.
- Missingness thresholds.
- Physical range checks.
- Chronological split prevents leakage.

Model tests:

- Predictions are finite.
- Forecast length matches requested horizon.
- Power predictions are clipped or flagged for physical bounds.
- Persistence baseline is always available.

CI gates:

- `ruff check`
- `ruff format --check`
- `mypy` or `pyright`
- `pytest`
- Docker build smoke test

## 13. Agile Delivery Plan

### Sprint 0: Foundation

Goal: create the repo skeleton and quality gates.

Deliverables:

- `pyproject.toml`
- package layout under `src/windenegy`
- config system
- CI workflow
- test fixtures
- README skeleton

Acceptance:

- CI passes on empty pipeline.
- `pytest` runs locally.
- `ruff` and type checks are wired.

### Sprint 1: Data Pipeline and Audit

Goal: make raw SCADA data trustworthy.

Deliverables:

- ingestion module
- schema normalization
- validation report
- chronological split
- first data audit notebook

Acceptance:

- Raw Kaggle CSV can be converted into processed parquet/csv.
- Validation report documents missingness, ranges, and anomalies.
- Data tests pass on fixtures.

### Sprint 2: Baselines

Goal: establish credible reference performance.

Deliverables:

- persistence baseline
- rolling mean baseline
- theoretical power curve evaluation
- gradient boosting baseline
- metrics report

Acceptance:

- Baseline metrics exist by horizon.
- No model result is reported without comparison to persistence.
- README includes preliminary baseline table.

### Sprint 3: Sequence Forecasting

Goal: add PatchTST or maintained sequence model.

Deliverables:

- sequence dataset generator
- PatchTST training script
- model artifact registry
- evaluation report vs baselines

Acceptance:

- Sequence model trains reproducibly.
- Saved artifact includes schema and metrics.
- Decision is documented if PatchTST does not beat baselines.

### Sprint 4: Uncertainty and Risk Views

Goal: make forecasts operationally useful.

Deliverables:

- conformal or quantile intervals
- ramp-event detector
- underproduction risk summary
- calibration report

Acceptance:

- Forecast response includes P10/P50/P90 or equivalent.
- Interval coverage is measured on validation/test data.
- Dashboard shows uncertainty bands.

### Sprint 5: API and Dashboard

Goal: ship a usable local product.

Deliverables:

- FastAPI service
- Streamlit dashboard
- Docker Compose
- OpenAPI examples

Acceptance:

- `docker compose up` starts API and dashboard.
- `/health`, `/metadata`, and `/forecast` work.
- Dashboard shows real forecast results from the API.

### Sprint 6: Polish and Portfolio Packaging

Goal: make it credible for Stellerus/job review.

Deliverables:

- final README
- architecture diagram
- model card
- data card
- screenshots
- deployment notes

Acceptance:

- A reviewer can understand the system in five minutes.
- Claims are tied to actual metrics.
- Stellerus relevance is explicit but not overstated.

## 14. Milestones

### MVP

- SCADA ingestion and validation.
- Persistence and gradient boosting baselines.
- Chronological evaluation.
- API serving baseline forecasts.
- Basic dashboard.

### Strong Portfolio Version

- Weather enrichment.
- PatchTST comparison.
- Uncertainty bands.
- Ramp-event analysis.
- Docker Compose.
- CI and model card.

### Stretch Version

- Satellite/weather feature placeholder interface.
- Multi-asset schema.
- MLflow or lightweight model registry.
- Live hosted demo.
- Batch forecast job.

## 15. Key Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dataset is too small/simple for PatchTST to win | Deep learning looks unjustified | Baselines first; document honest results |
| API feature drift from training | Incorrect forecasts | Shared feature pipeline and artifact schema |
| Weather enrichment is blocked by data access | Scope delay | Keep `NullWeatherProvider`; make weather optional |
| Dashboard becomes superficial | Weak product signal | Start with operational forecast and risk views |
| Overclaiming Stellerus fit | Credibility loss | Frame as small-scale analogue, not satellite production system |

## 16. Definition of Done

The project is done when:

- A new user can run tests and start the app from documented commands.
- The raw-data-to-model pipeline is reproducible.
- The API serves forecasts from a versioned model artifact.
- The dashboard uses the API, not static screenshots.
- Metrics compare all models against persistence.
- Forecast uncertainty or confidence is shown.
- README includes architecture, data assumptions, results, limitations, and Stellerus relevance.
- CI passes.

## 17. Recommended First Implementation Order

1. Create `pyproject.toml`, package layout, and CI.
2. Implement data ingestion, normalization, and validation.
3. Add fixture-based tests.
4. Implement persistence and power-curve baselines.
5. Add gradient boosting baseline.
6. Build evaluation reports.
7. Add API around the best baseline.
8. Add dashboard.
9. Add PatchTST and compare honestly.
10. Add uncertainty and Stellerus-focused polish.

