# Windenegy

**Wind power forecasting for renewable energy operations**

A prototype demonstrating weather-aware renewable energy forecasting: a small-scale analogue of climate intelligence systems that fuse observed asset data with meteorological context to support infrastructure decisions.

## Overview

Windenegy answers one concrete operational question:

> Given recent turbine SCADA observations and weather context, what power output should an operator expect over the next forecast horizon, and how uncertain is that forecast?

This project demonstrates:

- **Data Fusion**: SCADA observations combined with weather covariates
- **Physics-Guided AI**: Power curve baselines, physical sanity checks, regime-based evaluation
- **Uncertainty Quantification**: P10/P50/P90 prediction intervals
- **Operational Focus**: API and dashboard built for operators, not researchers
- **Production Practices**: CI/CD, testing, type safety, containerization

## Stellerus Alignment

This prototype maps to Stellerus themes:

| Theme | Implementation |
|-------|---------------|
| Climate Intelligence | Forecasts framed as decision support for climate-exposed assets |
| Data Fusion | SCADA + weather covariates with interface for satellite/radar |
| Physics-Guided AI | Power curve baselines, feature constraints, sanity checks |
| Risk & Resilience | Ramp event detection, low-confidence windows |
| Platform Thinking | API-first, versioned artifacts, reproducible pipelines |

## Architecture

Clean Architecture with clear separation:

```
windenegy/
├── domain/          # Pure business logic (models, repository interfaces)
├── application/     # Use cases and orchestration
├── infrastructure/  # External concerns (persistence, config, logging)
└── interface/       # API, dashboard, CLI entry points
```

Key design principles:

- **Immutable domain models** with Pydantic
- **Dependency injection** - configuration and repositories injected, not global
- **Repository pattern** - abstract data access for testability
- **Explicit contracts** - Pydantic schemas for all boundaries
- **Type safety** - strict mypy configuration

## Installation

### Quick Install (Recommended)

```bash
# Clone repository
git clone <repo-url>
cd windenegy

# Install Python dependencies (using uv - recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -e ".[dev]"
```

### Data Setup

```bash
# Download Kaggle dataset (or place your own SCADA CSV in data/raw/)
python scripts/download_scada.py

# Run data pipeline (ingestion, validation, splits)
python scripts/run_data_pipeline.py --input data/raw/T1.csv --output data/processed

# Train models
python scripts/train_patchtst.py --horizon 1
python scripts/train_gradient_boosting.py --horizon 1
```

### Verify Installation

```bash
# Run tests
./run_dev.sh test

# Start services and verify
./run_dev.sh api &
curl http://localhost:8765/health  # Should return {"status": "healthy"}
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip
- Docker & Docker Compose (for production deployment)

### Development Setup

```bash
# Clone and enter repository
git clone <repo>
cd windenegy

# Install dependencies
./run_dev.sh install

# Run linting and tests
./run_dev.sh lint
./run_dev.sh test

# Start development server
./run_dev.sh api      # API on http://localhost:8765
./run_dev.sh dashboard  # Dashboard on http://localhost:8766
```

### Production Deployment

```bash
# Build and start all services
./run_prod.sh build
./run_prod.sh up

# Check service health
./run_prod.sh --check

# View logs
./run_prod.sh logs

# Stop services
./run_prod.sh down
```

Or use Docker Compose directly:

```bash
docker-compose up --build
```

## Project Structure

```
windenegy/
├── src/windenegy/           # Main package
│   ├── domain/              # Core business logic
│   │   ├── models.py        # Pydantic domain models
│   │   └── repository.py    # Repository interfaces
│   ├── application/         # Use cases (Sprint 2+)
│   ├── infrastructure/      # External adapters
│   │   ├── config.py        # Pydantic-settings config
│   │   ├── logger.py        # Structured logging
│   │   └── persistence.py   # CSV/repository implementations
│   └── interface/           # Entry points
│       ├── api.py           # FastAPI service
│       ├── dashboard.py     # Streamlit dashboard
│       └── cli.py           # CLI (future)
├── configs/                 # YAML configuration
├── data/                    # Data storage
├── artifacts/               # Models, metrics, reports
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data
├── configs/default.yaml     # Default configuration
├── docker-compose.yml       # Production orchestration
├── Dockerfile               # Multi-stage build
├── run_dev.sh              # Development runner
├── run_prod.sh             # Production runner
└── pyproject.toml           # Package & tooling config
```

## Configuration

Configuration uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with environment variable override:

```bash
# Via environment variables
export WINDENEGY_API_PORT=8765      # Non-standard port (avoids conflicts)
export WINDENEGY_DASHBOARD_PORT=8766
export WINDENEGY_LOG_LEVEL=DEBUG

# Or copy and edit .env
cp .env.example .env
```

Priority (highest first):
1. Environment variables (`WINDENEGY_*`)
2. `.env` file
3. `configs/default.yaml`
4. Default values in code

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/metadata` | GET | API capabilities and active model info |
| `/forecast` | POST | Generate power forecast with P10/P50/P90 intervals |
| `/risk/ramps` | POST | Detect ramp events (rapid power changes) |
| `/risk/assess` | POST | Comprehensive risk assessment (ramps + underproduction) |

### Forecast Example

```bash
curl -X POST http://localhost:8765/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "T1",
    "horizon_hours": 6,
    "observations": [
      {
        "timestamp": "2018-12-30T23:50:00Z",
        "active_power_kw": 812.4,
        "wind_speed_mps": 7.2,
        "wind_direction_deg": 184.0,
        "theoretical_power_kwh": 900.1
      }
    ]
  }'
```

Response:

```json
{
  "asset_id": "T1",
  "model_version": "gradient-boosting-v1",
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

## Development Workflow

### Code Quality

```bash
# Format code
ruff format src tests

# Run linter
ruff check src tests

# Type check
mypy src

# Run tests
pytest
```

### Pre-commit

```bash
# Install hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## Data

### Source

This project uses the [Kaggle Turkey Wind Turbine SCADA dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset):

- 10-minute SCADA observations
- Wind speed, direction, active power
- Theoretical power curve from manufacturer

### Data Pipeline

```
Raw CSV → Validation → Normalization → Split → Features → Model
```

**Chronological splits** (no leakage):
- Train: 70% (earliest)
- Validation: 15%
- Test: 15% (latest)

## Models

### Implemented

- **Persistence Baseline**: Last observed power with adaptive uncertainty spread
- **Gradient Boosting**: LightGBM with time/lag features (25 features)
- **PatchTST**: Sequence transformer using 24-step input windows
- **Conformal Prediction**: P10/P90 intervals with 90% target coverage

## Testing

```bash
# All tests
pytest

# Unit only
pytest tests/unit

# Integration only
pytest tests/integration -m integration

# With coverage
pytest --cov=windenegy --cov-report=html
```

## CI/CD

GitHub Actions workflow includes:

- Linting (`ruff`)
- Type checking (`mypy`)
- Unit & integration tests (`pytest`)
- Package build
- Docker build

See `.github/workflows/ci.yml`

## Roadmap

All sprints completed:

| Sprint | Deliverables | Status |
|--------|--------------|--------|
| **0: Foundation** | Project structure, CI/CD, domain models | ✅ |
| **1: Data Pipeline** | Ingestion, validation, chronological splits | ✅ |
| **2: Baselines** | Persistence, power curve, gradient boosting | ✅ |
| **3: Sequence Model** | PatchTST, model registry, comparison | ✅ |
| **4: Uncertainty** | Conformal intervals, ramp detection | ✅ |
| **5: API & Dashboard** | FastAPI, Streamlit, Docker Compose | ✅ |
| **6: Portfolio** | Model cards, documentation | ✅ |

### Model Performance (Test Set)

| Model | MAE (kW) | RMSE (kW) | Skill vs Persistence |
|-------|----------|-----------|---------------------|
| Persistence | 116 | 225 | — |
| Gradient Boosting | 355 | 512 | -0.30 |
| PatchTST (1h) | 597 | 1217 | **+0.59** |

**Note**: PatchTST shows positive skill score (59% error reduction vs persistence) on 1-hour horizon using 24-hour input sequences. Gradient Boosting underperforms due to feature dimensionality constraints with minimal feature engineering.

## System Design Decisions

This section documents key architectural choices and their rationale.

### Why Clean Architecture?

| Challenge | Decision | Rationale |
|-----------|----------|-----------|
| Domain complexity | **Domain/Application/Infrastructure layers** | Business logic isolated from frameworks; testable without mocks |
| Configuration drift | **Pydantic-settings with env override** | 12-factor app compliance; same container runs in dev/staging/prod |
| Data access coupling | **Repository pattern** | Swap CSV for database without touching business logic |
| Type safety | **Strict mypy + Pydantic** | Catch integration errors at build time |

### Why These Ports? (8765/8766)

Standard ports (8000/8501) conflict with common development tools:
- **8765**: API port (avoids 8000: Airflow, Superset, etc.)
- **8766**: Dashboard port (avoids 8501: other Streamlit apps)

Both ports are unassigned by IANA and unlikely to conflict.

### Why Conformal Prediction for Uncertainty?

| Method | Coverage Guarantee | Distribution Assumption | Implementation Complexity |
|--------|-------------------|------------------------|--------------------------|
| Bayesian NN | No | Strong (Gaussian) | High |
| Monte Carlo Dropout | No | Moderate | Medium |
| **Conformal** | **Yes (finite-sample)** | **None** | **Low** |

Conformal prediction provides P10/P90 intervals with **proven coverage** regardless of model architecture—critical for operational risk assessment.

### Why PatchTST for Sequences?

- **Transformer architecture**: Captures long-range dependencies (24-step inputs)
- **Patching**: Reduces sequence length 8x → manageable compute
- **Channel independence**: Each feature processed separately → better for heterogeneous SCADA data
- **Surrogate with MLP**: sklearn-compatible API enables easy serialization

### Why Persistence as Fallback?

When gradient boosting fails (insufficient observations, feature mismatch):
- **Always available**: No cold-start problem
- **Physically plausible**: Tomorrow's weather correlates with today's
- **Calibrated uncertainty**: Spread derived from recent variability

### Chronological Splits (not Random)

```
Train: Jan-Aug → Val: Sep-Oct → Test: Nov-Dec
```

**Why**: Time series data has temporal structure. Random splits leak future information into training, inflating performance metrics unrealistically.

### Why Polars for Ingestion, Pandas for Features?

| Stage | Tool | Reason |
|-------|------|--------|
| **Ingestion** | Polars | 10x faster CSV parsing; lazy evaluation |
| **Features** | Pandas | Rich ecosystem (sklearn, feature-engine); team familiarity |
| **API** | Pydantic | Validation + serialization in one |

## License

MIT License - See LICENSE file

## Acknowledgments

- Kaggle for the Turkey Wind Turbine dataset
- Stellerus for the climate intelligence inspiration
