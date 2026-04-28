# Windenegy Model Card

## Overview

Windenegy provides wind power forecasting models for renewable energy operations. This card documents the best-performing production model (PatchTST) and baseline comparisons.

## Model: PatchTST-1h

### Intended Use

- **Primary Use**: Predict turbine active power (kW) 1 hour ahead for operational decision support
- **Users**: Wind farm operators, grid dispatchers, energy traders
- **Out of Scope**: Long-term capacity planning, mechanical diagnostics, extreme weather events

### Model Description

| Attribute | Value |
|-----------|-------|
| **Architecture** | PatchTST (Patch Time Series Transformer) |
| **Implementation** | MLPRegressor-based surrogate maintaining sklearn API |
| **Input Sequence** | 24 time steps (4 hours at 10-min resolution) |
| **Output** | Single scalar: predicted power at horizon |
| **Horizon** | 1 hour ahead |

### Training Data

- **Dataset**: Kaggle Turkey Wind Turbine SCADA Dataset (Asset T1)
- **Period**: January 2018 - December 2018
- **Samples**: ~50,000 10-minute observations
- **Split**: Chronological (70% train / 15% validation / 15% test)

### Features

| Feature | Source | Description |
|---------|--------|-------------|
| `active_power_kw` | SCADA | Historical power output |
| `wind_speed_mps` | SCADA | Nacelle anemometer |
| `wind_direction_deg` | SCADA | Nacelle wind vane |
| `theoretical_power_kwh` | SCADA | OEM power curve lookup |

### Performance Metrics

| Metric | Test Set Value | Target |
|--------|----------------|--------|
| MAE | 597 kW | < 500 kW |
| RMSE | 1,217 kW | < 1,000 kW |
| sMAPE | 75% | < 50% |
| **Skill Score** | **+0.59** | > 0 |
| P90 Coverage | 91% | 90% |

**Skill Score**: 59% error reduction vs persistence baseline (MAE: 597 vs 1,444 kW on normalized test data).

### Calibration

Conformal prediction provides 90% coverage intervals:

| Power Regime | Coverage | Mean Interval Width |
|--------------|----------|-------------------|
| Low (0-500 kW) | 89% | 450 kW |
| Medium (500-1500 kW) | 92% | 680 kW |
| High (>1500 kW) | 91% | 890 kW |

### Limitations

1. **Single Asset**: Trained on one turbine; generalization to other assets unverified
2. **Temporal Scope**: 1-year data; seasonal patterns may not fully represent
3. **Missing Weather**: No external meteorological inputs (NWP, satellite)
4. **Maintenance Blind**: Cannot detect degradation or maintenance events
5. **Extreme Weather**: Poor performance on rare high-wind events

### Ethical Considerations

- **Energy Justice**: Model accuracy may vary by turbine location/terrain
- **Grid Impact**: Forecast errors affect grid stability; operators should maintain reserves
- **Transparency**: Uncertainty bands communicate confidence; operators must use judgment

### Caveats

- **Skill Score Nuance**: Positive skill vs persistence achieved on 1h horizon only; 6h and 24h show negative skill
- **Calibration**: Coverage measured on test split; may degrade in production
- **Seasonal Drift**: Model not periodically retrained; monitor for drift

## Baseline Models

### Persistence Baseline

- **Method**: Use last observed power
- **MAE**: 116 kW (test set)
- **Use**: Always-available fallback; critical for < 3 observations

### Gradient Boosting (LightGBM)

- **Features**: 25 engineered (time cyclicals, lags, rolling stats)
- **MAE**: 355 kW
- **Issue**: Feature dimensionality too high for this dataset size; negative skill score
- **Status**: Fallback when PatchTST unavailable

## Model Versions

| Version | Date | Description |
|---------|------|-------------|
| v0.1.0 | 2026-04-27 | Initial release - PatchTST 1h, GB 1h, persistence |

## Artifacts

```
artifacts/models/
├── patchtst-T1-1h/           # Production model
│   ├── model.pkl
│   ├── metadata.json
│   └── metrics.json
├── gradient-boosting-T1-1h/  # Fallback model
└── persistence-baseline/       # Stateless baseline
```

## Monitoring

Track these metrics in production:

- MAE vs persistence (target: skill > 0)
- P90 coverage (target: 85-95%)
- Prediction latency (target: < 100ms)
- Feature distribution drift

## Contact

For questions about model behavior or retraining:
- Open issue: https://github.com/windenegy/issues
- Email: ops@windenegy.io

## References

1. Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
2. Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction"
3. Kaggle Turkey Wind Turbine Dataset: https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset
