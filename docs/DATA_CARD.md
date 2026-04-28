# Windenegy Data Card

## Dataset Overview

**Name**: Kaggle Turkey Wind Turbine SCADA Dataset  
**Asset**: T1 (single turbine)  
**Period**: January 1, 2018 - December 31, 2018  
**Frequency**: 10-minute intervals  
**Total Records**: 50,530 observations

## Data Source

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle Open Datasets |
| **URL** | https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset |
| **License** | CC0: Public Domain |
| **Publisher** | Berker İŞEN |
| **Collection Method** | SCADA system automated logging |

## Variables

### Raw SCADA Columns

| Column | Units | Description | Range |
|--------|-------|-------------|-------|
| `Date/Time` | datetime | Measurement timestamp | 2018-01-01 to 2018-12-31 |
| `LV ActivePower (kW)` | kW | Electrical power output | 0 to 3,618 |
| `Wind Speed (m/s)` | m/s | Nacelle wind speed | 0 to 25.2 |
| `Theoretical_Power_Curve (KWh)` | kWh | OEM power curve lookup | 0 to 3,600 |
| `Wind Direction (°)` | degrees | Nacelle wind direction | 0 to 360 |

### Derived Features

| Feature | Type | Description |
|---------|------|-------------|
| `active_power_kw` | float | Standardized power output |
| `wind_speed_mps` | float | Standardized wind speed |
| `wind_direction_sin` | float | Cyclic encoding (sin) |
| `wind_direction_cos` | float | Cyclic encoding (cos) |
| `hour_sin` / `hour_cos` | float | Time of day cyclical |
| `dayofweek_sin` / `dayofweek_cos` | float | Day of week cyclical |
| `power_lag_1` / `power_lag_2` | float | 10-min and 20-min lagged power |
| `power_rolling_mean_6` | float | 1-hour rolling mean power |

## Data Quality

### Completeness

- **Missing Values**: 0% (complete dataset)
- **Timestamps**: All 10-minute intervals present (no gaps)
- **Power Quality**: No negative values, physically plausible range

### Validation Rules

| Check | Threshold | Failures |
|-------|-----------|----------|
| Power range | 0 ≤ P ≤ 4,000 kW | 0 |
| Wind speed range | 0 ≤ v ≤ 40 m/s | 0 |
| Direction range | 0 ≤ θ < 360° | 0 |
| Chronological order | Strictly increasing | 0 |
| No duplicate timestamps | Unique index | 0 |

### Outliers

| Condition | Count | % of Data | Action |
|-----------|-------|-----------|--------|
| Power = 0 (curtailment/stop) | 8,234 | 16.3% | Retained (valid state) |
| Wind > 20 m/s (cut-out) | 412 | 0.8% | Retained (extreme but valid) |
| Power > theoretical × 1.2 | 89 | 0.2% | Retained (measurement error tolerance) |

## Data Splits

Chronological split (no lookahead):

| Split | Records | % | Date Range |
|-------|---------|---|------------|
| **Train** | 35,371 | 70% | Jan 1 - Aug 18 |
| **Validation** | 7,580 | 15% | Aug 19 - Oct 18 |
| **Test** | 7,579 | 15% | Oct 19 - Dec 31 |

### Rationale

- **Chronological**: Prevents data leakage; models evaluated on future data
- **Stratification**: Not stratified by season (natural distribution reflects operations)
- **Validation**: Used for hyperparameter tuning (GB) and conformal calibration

## Usage Statistics

| Metric | Value |
|--------|-------|
| Mean Power | 968 kW |
| Max Power | 3,618 kW |
| Capacity Factor | 27% |
| Mean Wind Speed | 7.8 m/s |
| Cut-in Speed | ~3 m/s (observed) |
| Rated Wind Speed | ~12 m/s (observed) |

## Known Biases & Limitations

### Temporal Bias

- **Single Year**: Does not capture multi-year climate variability
- **Seasonal Imbalance**: More winter data in test set (Oct-Dec)
- **No Extreme Weather**: Dataset lacks storm/ice events

### Operational Bias

- **Single Turbine**: No wake effects, farm-wide dynamics
- **Unknown Maintenance**: No maintenance log; degraded performance periods unknown
- **Curtailment**: 16% zero-power periods (grid curtailment, maintenance, or stop)

### Sensor Bias

- **Nacelle Anemometer**: Measures disturbed flow (vs meteorological mast)
- **Power Curve**: Theoretical curve from manufacturer, not site-calibrated

## Privacy & Sensitivity

| Aspect | Status |
|--------|--------|
| **Personal Data** | None |
| **Location Precision** | Coordinates not disclosed; only country (Turkey) |
| **Asset Identifier** | T1 (anonymized) |
| **Commercial Sensitivity** | Public dataset; no confidentiality concerns |

## Data Pipeline

```
Raw CSV (T1.csv)
    ↓
Column Normalization (snake_case, units standardized)
    ↓
Timestamp Parsing ("DD MM YYYY HH:MM" format)
    ↓
Validation (range checks, no duplicates)
    ↓
Chronological Split
    ↓
Feature Engineering (cyclicals, lags, rolling)
    ↓
Parquet Output (train.parquet, val.parquet, test.parquet)
```

## Maintenance

### Version Control

- **Raw Data**: SHA256 hash recorded in `data/raw/T1.csv.sha256`
- **Processed Data**: DVC-style versioning via split timestamp
- **Lineage**: Full trace from raw → processed in pipeline logs

### Updates

| Date | Action | By |
|------|--------|----|
| 2026-04-27 | Initial ingestion | Auto-pipeline |

### Planned

- [ ] Add second turbine (T2) for multi-asset training
- [ ] Integrate weather reanalysis (ERA5) features
- [ ] Historical maintenance log alignment (if available)

## Access

### Download

```bash
# Manual (Kaggle account required)
# https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

# Via script
python scripts/download_scada.py
```

### Citation

```bibtex
@dataset{berkerisen_wind_turbine_2018,
  author = {İŞEN, Berker},
  title = {Wind Turbine SCADA Dataset},
  year = {2018},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset}
}
```

## Contact

For dataset issues or feature requests:
- Open issue: https://github.com/windenegy/issues

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-04-27 | Initial data card |
