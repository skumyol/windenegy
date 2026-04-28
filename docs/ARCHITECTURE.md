# Windenegy Architecture

## Overview

Clean Architecture with four layers and explicit dependency direction (inward-pointing arrows).

```
┌─────────────────────────────────────────────────────────────┐
│  Interface Layer    │  FastAPI, Streamlit, CLI             │
│  (adapters)         │  HTTP/Web entry points               │
├─────────────────────────────────────────────────────────────┤
│  Application Layer  │  ForecastService, Training, Risk     │
│  (use cases)      │  Orchestration, feature engineering    │
├─────────────────────────────────────────────────────────────┤
│  Domain Layer     │  Models, Repository interfaces       │
│  (business rules) │  Immutable Pydantic schemas            │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure     │  Polars/pandas IO, sklearn models    │
│  Layer             │  Config, logging, persistence        │
└─────────────────────────────────────────────────────────────┘
         ↑ Dependencies flow inward (domain knows nothing of infrastructure)
```

## Layers

### Domain (`src/windenegy/domain/`)

**Rule**: No external dependencies. Pure business logic.

| Module | Purpose |
|--------|---------|
| `models.py` | Pydantic domain models: `PowerForecast`, `TurbineObservation`, `ForecastPoint` |
| `repository.py` | Repository interfaces: `TurbineRepository`, `AssetRepository` |
| `sequence.py` | Protocols for sequence models: `SequenceModel`, `SequenceConfig` |

Key invariants:
- `TurbineObservation.timestamp` is timezone-aware UTC
- `ForecastPoint.p10 ≤ p50 ≤ p90`
- `PowerForecast.horizon_hours` ∈ {1, 6, 24}

### Application (`src/windenegy/application/`)

**Rule**: Depends only on domain. Orchestrates use cases.

| Module | Purpose |
|--------|---------|
| `forecasting.py` | `ForecastService` - persistence baseline with uncertainty |
| `baseline.py` | Persistence, power curve, rolling mean baselines |
| `training.py` | `GradientBoostingPowerModel` - training & artifact management |
| `features.py` | Feature engineering: cyclicals, lags, rolling stats |
| `evaluation.py` | Metrics: MAE, RMSE, sMAPE, skill score, coverage |
| `sequence_data.py` | `SequenceDatasetBuilder` - sliding window sequences |
| `uncertainty.py` | `ConformalPredictor` - P10/P90 prediction intervals |
| `risk.py` | `RampDetector`, `UnderproductionAnalyzer` |
| `calibration_report.py` | Coverage/sharpness evaluation by power regime |

### Infrastructure (`src/windenegy/infrastructure/`)

**Rule**: Implements domain interfaces. Houses external concerns.

| Module | Purpose |
|--------|---------|
| `config.py` | Pydantic-settings with YAML/env var override |
| `logger.py` | Structured logging with rotation |
| `persistence.py` | Parquet/CSV repository implementations |
| `patchtst_model.py` | `PatchTSTModel` - sklearn-compatible transformer surrogate |

### Interface (`src/windenegy/interface/`)

**Rule**: Adapts application for external protocols.

| Module | Purpose | Port |
|--------|---------|------|
| `api.py` | FastAPI service | 8765 |
| `dashboard.py` | Streamlit dashboard | 8766 |

## Data Flow

### Forecast Request

```
POST /forecast
    ↓
FastAPI validates Pydantic schema
    ↓
ForecastService.forecast(observations, horizon)
    ├─ If trained model matches horizon:
    │   GradientBoostingPowerModel.predict()
    │   ConformalPredictor.predict_interval()
    └─ Else:
    │   Persistence baseline (last power ± adaptive spread)
    ↓
PowerForecast (asset_id, points[], warnings[])
    ↓
JSON response with P10/P50/P90
```

### Training Pipeline

```
Raw CSV (data/raw/T1.csv)
    ↓
DataIngestionService.ingest_raw_csv()
    ├─ Timestamp parsing: "%d %m %Y %H:%M"
    ├─ Column normalization (snake_case)
    └─ Validation (range checks, no duplicates)
    ↓
Chronological Split (70/15/15)
    ↓
Feature Engineering
    ├─ Cyclicals: hour, dayofweek, wind direction
    ├─ Lags: power 10-min and 20-min ago
    └─ Rolling: 1-hour mean power
    ↓
Model Training
    ├─ Gradient Boosting (LightGBM, 25 features)
    ├─ PatchTST (24-step sequences)
    └─ Conformal Calibration (validation split)
    ↓
Artifact Save (artifacts/models/)
    ├─ model.pkl
    ├─ metadata.json
    └─ metrics.json
```

## Dependency Injection

Configuration and repositories are injected, not global:

```python
# Domain defines interface
class TurbineRepository(Protocol):
    def load(self, asset_id: str) -> list[TurbineObservation]: ...

# Infrastructure implements it
class ParquetTurbineRepository:
    def __init__(self, data_dir: Path) -> None: ...

# Application uses interface
class ForecastService:
    def __init__(self, repository: TurbineRepository) -> None: ...

# Interface wires concrete implementations
repository = ParquetTurbineRepository(config.data_path)
service = ForecastService(repository)
```

## Error Handling

| Layer | Strategy |
|-------|----------|
| Domain | ValueError with descriptive messages |
| Application | Domain exceptions propagate; add context |
| Interface | HTTP 400 for client errors, 422 for validation |

## Testing Strategy

| Type | Scope | Tools |
|------|-------|-------|
| Unit | Domain models, pure functions | pytest |
| Integration | Repository + application | pytest + fixtures |
| API | Endpoint contracts | requests + pytest |
| E2E | Dashboard flows | Playwright (future) |

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.11 |
| Data | Polars (ingestion), pandas (features), numpy |
| ML | scikit-learn, lightgbm |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Config | Pydantic-settings, YAML |
| Testing | pytest, pytest-cov |
| Linting | ruff, mypy |
| Packaging | uv, pyproject.toml |
| Container | Docker, Docker Compose |

## Ports

| Service | Dev | Prod | Rationale |
|---------|-----|------|-----------|
| API | 8765 | 8765 | Non-standard; avoids 8000 conflicts |
| Dashboard | 8766 | 8766 | Non-standard; avoids 8501 conflicts |

## Deployment

```yaml
# docker-compose.yml
api:
  build: .
  ports: ["8765:8765"]
  volumes:
    - ./data:/app/data:ro
    - ./artifacts:/app/artifacts

dashboard:
  build: .
  ports: ["8766:8766"]
  environment:
    WINDENEGY_DASHBOARD_API_URL: http://api:8765
  depends_on:
    api: { condition: service_healthy }
```

## Monitoring (Planned)

| Metric | Target | Method |
|--------|--------|--------|
| Forecast latency | < 100ms | FastAPI middleware |
| Coverage drift | 85-95% | Conformal predictor logs |
| MAE vs persistence | skill > 0 | nightly batch evaluation |
| Feature drift | KS test p > 0.05 | weekly distribution check |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04 | Pydantic over dataclasses | Validation at boundaries, JSON serialization |
| 2026-04 | Polars for ingestion | 10x faster CSV parsing than pandas |
| 2026-04 | sklearn API for PatchTST | Easier serialization, compatibility with existing tooling |
| 2026-04 | Conformal prediction | Distribution-free, finite-sample coverage guarantees |
| 2026-04 | Persistence as fallback | Always available, no cold-start problem |
| 2026-04 | Chronological splits | Prevents temporal leakage in time series |
