#!/usr/bin/env python3
"""Train and evaluate every forecasting model on the same chronological split.

This is the single orchestrator for methodological comparison:

- Builds one chronological train/val/test split.
- For every horizon in {1h, 6h, 24h}, runs every model:
  Persistence, PowerCurve, RollingMean, GradientBoosting, PatchTST.
- Writes one test-output CSV per (model, horizon) with the SAME schema
  so the dashboard can overlay every model on top of the actuals.
- Writes ``model_comparison.json`` aggregating metrics for the whole grid.
- Persists ML model artifacts so the API can serve the gradient
  boosting model.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --horizons 1 6
    python scripts/train_all.py --csv data/raw/T1.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from windenegy.application.comparison import (  # noqa: E402
    SplitInfo,
    evaluate_gradient_boosting,
    evaluate_patchtst,
    evaluate_persistence,
    evaluate_power_curve,
    evaluate_rolling_mean,
    metrics_from_outputs,
    prepare_split,
    save_metrics_json,
    save_split_json,
    save_test_outputs,
)
from windenegy.application.training import GradientBoostingPowerModel  # noqa: E402
from windenegy.domain.models import ModelArtifactMetadata  # noqa: E402
from windenegy.infrastructure.persistence import FileSystemModelRepository  # noqa: E402

DEFAULT_DATA_PATH = Path("data/raw/T1.csv")
DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_METRICS_DIR = Path("artifacts/metrics")
DEFAULT_HORIZONS: tuple[int, ...] = (1, 6, 24)


@dataclass
class RunSummary:
    """One row in the unified comparison table."""

    model_id: str
    model_type: str
    horizon_hours: int
    mae: float
    rmse: float
    mape: float
    skill_score: float | None
    coverage_p90: float | None
    test_rows: int
    test_outputs_path: str
    model_artifact_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "horizon_hours": self.horizon_hours,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "skill_score": (
                round(self.skill_score, 4) if self.skill_score is not None else None
            ),
            "coverage_p90": (
                round(self.coverage_p90, 4) if self.coverage_p90 is not None else None
            ),
            "test_rows": self.test_rows,
            "test_outputs_path": self.test_outputs_path,
            "model_artifact_path": self.model_artifact_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summary(
    model_id: str,
    model_type: str,
    horizon_hours: int,
    out_df: Any,
    metrics_dir: Path,
    artifact_path: Path | None,
) -> RunSummary:
    metrics = metrics_from_outputs(out_df, model_id=model_id, horizon_hours=horizon_hours)
    csv_path = save_test_outputs(metrics_dir, model_id, out_df)
    save_metrics_json(metrics_dir, model_id, metrics)
    return RunSummary(
        model_id=model_id,
        model_type=model_type,
        horizon_hours=horizon_hours,
        mae=metrics.mae,
        rmse=metrics.rmse,
        mape=metrics.mape,
        skill_score=metrics.skill_score,
        coverage_p90=metrics.coverage_p90,
        test_rows=int(len(out_df)),
        test_outputs_path=str(csv_path),
        model_artifact_path=str(artifact_path) if artifact_path else None,
    )


def _save_gradient_boosting_artifact(
    model_dir: Path,
    model_id: str,
    horizon_hours: int,
    artifact: dict[str, Any],
    regressor: Any,
    feature_columns: list[str],
    metrics: Any,
    split: SplitInfo,
) -> Path:
    """Persist a GBM artifact in the format the API expects."""
    model = GradientBoostingPowerModel(
        regressor=regressor,
        feature_columns=feature_columns,
        residual_p90=float(artifact["residual_p90"]),
        bias_correction=float(artifact["bias_correction"]),
        horizon_hours=horizon_hours,
        model_version=model_id,
        lags=tuple(artifact["lags"]),
        windows=tuple(artifact["windows"]),
        predicts_delta=True,
    )
    metadata = ModelArtifactMetadata(
        model_type="gradient_boosting",
        model_version=model_id,
        feature_schema={col: "float" for col in feature_columns},
        target="active_power_kw",
        horizon_hours=horizon_hours,
        metrics=metrics,
        config_snapshot={
            "asset_id": "T1",
            "rows_total": split.n_total,
            "training_rows": artifact["n_train"],
            "validation_rows": artifact["n_val"],
            "test_rows": artifact["n_test"],
            "bias_correction": artifact["bias_correction"],
            "step_minutes": split.step_minutes,
            "horizon_steps": artifact["horizon_steps"],
            "lags": artifact["lags"],
            "windows": artifact["windows"],
            "train_end_ts": split.train_end_ts.isoformat(),
            "val_end_ts": split.val_end_ts.isoformat(),
            "test_end_ts": split.test_end_ts.isoformat(),
        },
    )
    repo = FileSystemModelRepository(model_dir)
    return repo.save_model(model_id, model, metadata)


def _save_patchtst_artifact(
    model_dir: Path,
    model_id: str,
    horizon_hours: int,
    artifact: dict[str, Any],
    internals: tuple[Any, Any],
    metrics: Any,
    split: SplitInfo,
) -> Path:
    """Persist a minimal PatchTST artifact (regressor + scaler + config)."""
    target_dir = model_dir / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    regressor, scaler = internals
    payload = {
        "kind": "patchtst_mlp_v1",
        "regressor": regressor,
        "scaler": scaler,
        "feature_columns": artifact["feature_columns"],
        "seq_len": artifact["seq_len"],
        "pred_len": artifact["pred_len"],
        "residual_p90": artifact["residual_p90"],
    }
    with (target_dir / "model.pkl").open("wb") as f:
        pickle.dump(payload, f)

    metadata = ModelArtifactMetadata(
        model_type="patchtst",
        model_version=model_id,
        feature_schema={c: "float" for c in artifact["feature_columns"]},
        target="active_power_kw",
        horizon_hours=horizon_hours,
        training_window_hours=int(artifact["seq_len"] * split.step_minutes / 60),
        metrics=metrics,
        config_snapshot={
            "seq_len": artifact["seq_len"],
            "pred_len": artifact["pred_len"],
            "n_features": artifact["n_features"],
            "n_train": artifact["n_train"],
            "n_val": artifact["n_val"],
            "n_test": artifact["n_test"],
            "residual_p90": artifact["residual_p90"],
        },
    )
    with (target_dir / "metadata.json").open("w") as f:
        json.dump(metadata.to_dict(), f, indent=2, default=str)
    return target_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    csv_path: Path,
    model_dir: Path,
    metrics_dir: Path,
    horizons: tuple[int, ...],
    train_ratio: float,
    val_ratio: float,
    skip_patchtst: bool,
) -> list[RunSummary]:
    """Run the full comparison grid and return summaries."""
    if not csv_path.exists():
        msg = f"SCADA CSV not found at {csv_path}"
        raise FileNotFoundError(msg)

    print(f"Loading and splitting {csv_path}...")
    split = prepare_split(csv_path, train_ratio=train_ratio, val_ratio=val_ratio)
    save_split_json(metrics_dir, split)
    print(
        f"  rows: {split.n_total:,} | train: {split.n_train:,} "
        f"| val: {split.n_val:,} | test: {split.n_test:,}"
    )
    print(f"  step: {split.step_minutes:.1f} min")
    print(f"  train end:  {split.train_end_ts}")
    print(f"  val end:    {split.val_end_ts}")
    print(f"  test end:   {split.test_end_ts}")

    asset_id = "T1"
    summaries: list[RunSummary] = []

    for horizon_hours in horizons:
        horizon_steps = max(1, round(horizon_hours * 60 / split.step_minutes))
        print(f"\n=== Horizon {horizon_hours}h  ({horizon_steps} steps) ===")

        # Persistence
        try:
            mid = f"persistence-{asset_id}-{horizon_hours}h"
            out, _ = evaluate_persistence(split, horizon_steps, horizon_hours, mid)
            summaries.append(_summary(mid, "persistence", horizon_hours, out, metrics_dir, None))
            print(f"  persistence: rows={len(out)}  mae={summaries[-1].mae:.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  persistence FAILED: {exc}")

        # Power curve
        try:
            mid = f"power-curve-{asset_id}-{horizon_hours}h"
            out, _ = evaluate_power_curve(split, horizon_steps, horizon_hours, mid)
            summaries.append(_summary(mid, "power_curve", horizon_hours, out, metrics_dir, None))
            print(f"  power_curve: rows={len(out)}  mae={summaries[-1].mae:.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  power_curve FAILED: {exc}")

        # Rolling mean
        try:
            mid = f"rolling-mean-{asset_id}-{horizon_hours}h"
            out, _ = evaluate_rolling_mean(split, horizon_steps, horizon_hours, mid)
            summaries.append(
                _summary(mid, "rolling_mean", horizon_hours, out, metrics_dir, None)
            )
            print(f"  rolling_mean: rows={len(out)}  mae={summaries[-1].mae:.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  rolling_mean FAILED: {exc}")

        # Gradient boosting
        try:
            mid = f"gradient-boosting-{asset_id}-{horizon_hours}h"
            out, artifact, regressor, feat_cols = evaluate_gradient_boosting(
                split, horizon_steps, horizon_hours, mid
            )
            metrics = metrics_from_outputs(out, mid, horizon_hours)
            artifact_path = _save_gradient_boosting_artifact(
                model_dir, mid, horizon_hours, artifact, regressor, feat_cols, metrics, split
            )
            csv_path_out = save_test_outputs(metrics_dir, mid, out)
            save_metrics_json(metrics_dir, mid, metrics)
            summaries.append(
                RunSummary(
                    model_id=mid,
                    model_type="gradient_boosting",
                    horizon_hours=horizon_hours,
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    mape=metrics.mape,
                    skill_score=metrics.skill_score,
                    coverage_p90=metrics.coverage_p90,
                    test_rows=int(len(out)),
                    test_outputs_path=str(csv_path_out),
                    model_artifact_path=str(artifact_path),
                )
            )
            print(
                f"  gradient_boosting: rows={len(out)}  "
                f"mae={metrics.mae:.1f}  skill={metrics.skill_score:+.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  gradient_boosting FAILED: {exc}")

        # PatchTST (optional, skipped on tiny datasets)
        if not skip_patchtst:
            try:
                mid = f"patchtst-{asset_id}-{horizon_hours}h"
                result = evaluate_patchtst(split, horizon_steps, horizon_hours, mid)
                if result is None:
                    print("  patchtst: skipped (not enough data)")
                else:
                    out, artifact, internals = result
                    metrics = metrics_from_outputs(out, mid, horizon_hours)
                    artifact_path = _save_patchtst_artifact(
                        model_dir, mid, horizon_hours, artifact, internals, metrics, split
                    )
                    csv_path_out = save_test_outputs(metrics_dir, mid, out)
                    save_metrics_json(metrics_dir, mid, metrics)
                    summaries.append(
                        RunSummary(
                            model_id=mid,
                            model_type="patchtst",
                            horizon_hours=horizon_hours,
                            mae=metrics.mae,
                            rmse=metrics.rmse,
                            mape=metrics.mape,
                            skill_score=metrics.skill_score,
                            coverage_p90=metrics.coverage_p90,
                            test_rows=int(len(out)),
                            test_outputs_path=str(csv_path_out),
                            model_artifact_path=str(artifact_path),
                        )
                    )
                    print(
                        f"  patchtst: rows={len(out)}  "
                        f"mae={metrics.mae:.1f}  skill={metrics.skill_score:+.3f}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  patchtst FAILED: {exc}")

    # Write unified comparison file
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": split.to_dict(),
        "horizons": list(horizons),
        "results": [s.to_dict() for s in summaries],
    }
    with (metrics_dir / "model_comparison.json").open("w") as f:
        json.dump(comparison_payload, f, indent=2, default=str)
    print(f"\nWrote {metrics_dir / 'model_comparison.json'}")

    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all models on the same split")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Forecast horizons in hours",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--skip-patchtst",
        action="store_true",
        help="Skip the PatchTST stand-in (useful for fast iteration)",
    )
    args = parser.parse_args()

    summaries = run(
        csv_path=args.csv,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        horizons=tuple(args.horizons),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        skip_patchtst=args.skip_patchtst,
    )

    if not summaries:
        print("No models trained successfully.")
        return 1

    print("\n" + "=" * 70)
    print(f"{'model':<32} {'h':>4} {'MAE':>9} {'RMSE':>9} {'skill':>8} {'cov':>7}")
    print("-" * 70)
    for s in summaries:
        skill = f"{s.skill_score:+.3f}" if s.skill_score is not None else "  n/a"
        cov = f"{s.coverage_p90 * 100:.1f}%" if s.coverage_p90 is not None else "  n/a"
        print(
            f"{s.model_type:<32} {s.horizon_hours:>4d} "
            f"{s.mae:>9.2f} {s.rmse:>9.2f} {skill:>8} {cov:>7}"
        )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
