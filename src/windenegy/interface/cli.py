"""Command-line entry point for local Windenegy workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from windenegy.application.features import normalize_scada_frame
from windenegy.application.training import train_gradient_boosting_from_csv


def main() -> None:
    """Run the Windenegy command-line interface."""
    parser = argparse.ArgumentParser(prog="windenegy")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="Validate a raw SCADA CSV")
    validate.add_argument("csv_path", type=Path)

    train = subcommands.add_parser("train", help="Train a gradient boosting forecaster")
    train.add_argument("csv_path", type=Path)
    train.add_argument("--asset-id", default="T1")
    train.add_argument("--horizon-hours", type=int, default=1)
    train.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    train.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))

    args = parser.parse_args()
    if args.command == "validate":
        frame = normalize_scada_frame(pd.read_csv(args.csv_path))
        summary = {
            "records": len(frame),
            "start": frame["timestamp"].min().isoformat(),
            "end": frame["timestamp"].max().isoformat(),
            "missing": frame.isna().sum().astype(int).to_dict(),
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "train":
        result = train_gradient_boosting_from_csv(
            csv_path=args.csv_path,
            model_dir=args.model_dir,
            metrics_dir=args.metrics_dir,
            horizon_hours=args.horizon_hours,
            asset_id=args.asset_id,
        )
        print(
            json.dumps(
                {
                    "model_id": result.model_id,
                    "model_path": str(result.model_path),
                    "metrics_path": str(result.metrics_path),
                    "metrics": (
                        result.metadata.metrics.to_dict()
                        if result.metadata.metrics is not None
                        else None
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
