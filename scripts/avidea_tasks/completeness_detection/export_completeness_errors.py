#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "path",
    "true_label",
    "predicted_label",
    "probability_incomplete",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export misclassified completeness images into folders."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-correct-borderline",
        type=int,
        default=0,
        help="Also export N correctly classified images nearest the threshold from each class.",
    )
    return parser.parse_args()


def safe_filename(source: Path, index: int) -> str:
    return f"{index:04d}__{source.name}"


def export_row(row: pd.Series, destination: Path, index: int):
    source = Path(str(row["path"]))
    if not source.exists():
        print(f"[WARNING] Missing image: {source}")
        return None

    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / safe_filename(source, index)
    shutil.copy2(source, output_path)

    return {
        "exported_path": str(output_path),
        "original_path": str(source),
        "true_label": str(row["true_label"]),
        "predicted_label": str(row["predicted_label"]),
        "probability_incomplete": float(row["probability_incomplete"]),
        "probability_threshold": (
            float(row["probability_threshold"])
            if "probability_threshold" in row and pd.notna(row["probability_threshold"])
            else ""
        ),
        "correct": bool(row["correct"]) if "correct" in row else (
            str(row["true_label"]) == str(row["predicted_label"])
        ),
    }


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "correct" not in df.columns:
        df["correct"] = df["true_label"] == df["predicted_label"]

    false_complete_rejections = df[
        (df["true_label"] == "complete")
        & (df["predicted_label"] == "incomplete")
    ].copy().sort_values("probability_incomplete", ascending=False)

    missed_incomplete = df[
        (df["true_label"] == "incomplete")
        & (df["predicted_label"] == "complete")
    ].copy().sort_values("probability_incomplete", ascending=True)

    manifest_rows = []

    fp_dir = args.output / "false_complete_rejections" / "true_complete_pred_incomplete"
    for index, (_, row) in enumerate(false_complete_rejections.iterrows(), start=1):
        result = export_row(row, fp_dir, index)
        if result is not None:
            result["error_type"] = "false_complete_rejection"
            manifest_rows.append(result)

    fn_dir = args.output / "missed_incomplete" / "true_incomplete_pred_complete"
    for index, (_, row) in enumerate(missed_incomplete.iterrows(), start=1):
        result = export_row(row, fn_dir, index)
        if result is not None:
            result["error_type"] = "missed_incomplete"
            manifest_rows.append(result)

    if args.copy_correct_borderline > 0:
        if "probability_threshold" not in df.columns:
            raise ValueError("Predictions CSV has no probability_threshold column.")

        thresholds = df["probability_threshold"].dropna().unique()
        if len(thresholds) == 0:
            raise ValueError("No probability threshold found in predictions CSV.")
        threshold = float(thresholds[0])

        for true_label, folder_name, error_type in [
            ("complete", "true_complete_pred_complete", "borderline_correct_complete"),
            ("incomplete", "true_incomplete_pred_incomplete", "borderline_correct_incomplete"),
        ]:
            subset = df[(df["correct"]) & (df["true_label"] == true_label)].copy()
            subset["distance_to_threshold"] = (
                subset["probability_incomplete"] - threshold
            ).abs()
            subset = subset.sort_values("distance_to_threshold").head(
                args.copy_correct_borderline
            )
            dest = args.output / "borderline_correct" / folder_name
            for index, (_, row) in enumerate(subset.iterrows(), start=1):
                result = export_row(row, dest, index)
                if result is not None:
                    result["error_type"] = error_type
                    manifest_rows.append(result)

    manifest_path = args.output / "errors_manifest.csv"
    if manifest_rows:
        fieldnames = [
            "error_type",
            "exported_path",
            "original_path",
            "true_label",
            "predicted_label",
            "probability_incomplete",
            "probability_threshold",
            "correct",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("Export complete")
    print(f"False complete rejections: {len(false_complete_rejections)}")
    print(f"Missed incomplete images:  {len(missed_incomplete)}")
    print(f"Output:                    {args.output}")
    print(f"Manifest:                  {manifest_path}")


if __name__ == "__main__":
    main()
