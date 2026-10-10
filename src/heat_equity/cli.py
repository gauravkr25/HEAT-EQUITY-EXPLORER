"""Command-line entry point for local project workflows."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def find_project_root(config_path: Path) -> Path:
    """Find the project root containing the given configuration file."""
    resolved_path = config_path.resolve()
    for parent in resolved_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not find pyproject.toml above config file: {config_path}")


def read_config(config_path: Path) -> dict[str, Any]:
    """Read a TOML configuration file."""
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def validate_inputs(config_path: Path) -> tuple[list[str], list[str]]:
    """Return missing required and optional input descriptions."""
    config = read_config(config_path)
    project_root = find_project_root(config_path)
    missing_required: list[str] = []
    missing_optional: list[str] = []

    inputs = config.get("inputs", [])
    if not isinstance(inputs, list):
        raise TypeError("The config must define inputs as an array of [[inputs]] tables.")

    for item in inputs:
        if not isinstance(item, dict) or "name" not in item or "path" not in item:
            raise ValueError("Every [[inputs]] entry must include name and path fields.")
        relative_path = Path(str(item["path"]))
        input_path = relative_path if relative_path.is_absolute() else project_root / relative_path
        if input_path.exists():
            continue

        description = f"{item['name']}: {input_path}"
        if item.get("required", True):
            missing_required.append(description)
        else:
            missing_optional.append(description)

    return missing_required, missing_optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heat-equity")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser(
        "validate-inputs", help="Check that configured manually downloaded data files exist."
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phoenix-2021.toml"),
        help="Path to the project TOML configuration.",
    )
    prepare_parser = commands.add_parser(
        "prepare", help="Prepare configured Landsat, MRLC, and tract data locally."
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phoenix-2021.toml"),
        help="Path to the project TOML configuration.",
    )
    prepare_parser.add_argument(
        "--minimum-valid-fraction", type=float, default=0.5,
        help="Minimum valid Landsat pixel fraction per grid cell.",
    )
    prepare_parser.add_argument(
        "--block-size-meters", type=int, default=2000,
        help="Spatial block width used for train/test separation.",
    )
    prepare_parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    train_parser = commands.add_parser(
        "train-table", help="Compare classical regressors using a prepared CSV or Parquet table."
    )
    train_parser.add_argument("table", type=Path, help="Prepared feature table in CSV format.")
    train_parser.add_argument(
        "--target", default="lst_celsius", help="Name of the numeric target column."
    )
    train_parser.add_argument("--split", default="split", help="Name of the train/test split column.")
    train_parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional columns that must not be used as model predictors.",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/latest"),
        help="Directory for models, metrics, and held-out predictions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        try:
            from heat_equity.data.prepare import prepare_scene

            config = read_config(args.config)
            project_root = find_project_root(args.config)
            outputs = prepare_scene(
                config,
                project_root,
                minimum_valid_fraction=args.minimum_valid_fraction,
                block_size_meters=args.block_size_meters,
                random_seed=args.seed,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"Data preparation could not run: {error}", file=sys.stderr)
            return 2
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-table":
        try:
            import pandas as pd

            from heat_equity.modeling.train import compare_regressors, save_model_runs

            if args.table.suffix.lower() == ".parquet":
                table = pd.read_parquet(args.table)
            else:
                table = pd.read_csv(args.table)
            runs = compare_regressors(
                table,
                target_column=args.target,
                split_column=args.split,
                excluded_columns=set(args.exclude),
            )
            save_model_runs(
                runs,
                args.output,
                model_table=table,
                target_column=args.target,
                split_column=args.split,
            )
        except (OSError, ValueError, KeyError) as error:
            print(f"Model training could not run: {error}", file=sys.stderr)
            return 2
        print(f"Saved {len(runs)} model runs to {args.output.resolve()}")
        print(json.dumps({name: run.metrics for name, run in runs.items()}, indent=2))
        return 0

    try:
        missing_required, missing_optional = validate_inputs(args.config)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"Input validation could not run: {error}", file=sys.stderr)
        return 2

    if missing_required:
        print(f"Missing {len(missing_required)} required input(s):")
        for description in missing_required:
            print(f"  - {description}")
    else:
        print("All required inputs are present.")

    if missing_optional:
        print(f"Missing {len(missing_optional)} optional input(s):")
        for description in missing_optional:
            print(f"  - {description}")

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())