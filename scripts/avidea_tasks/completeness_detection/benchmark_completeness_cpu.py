#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# Apply CPU thread limits before importing native ML libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import psutil
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision import models, transforms
from tqdm import tqdm


VIEWS = ["front", "back", "left", "right"]
CLASS_NAMES = ["complete", "incomplete"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_THRESHOLDS = {
    "front": 0.65,
    "back": 0.73,
    "left": 0.62,
    "right": 0.54,
}

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU benchmark for four per-view MobileNetV3-Small vehicle "
            "completeness classifiers."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Root of the fixed split dataset. The script expects "
            "DATASET/test/{front,back,left,right}/{complete,incomplete}."
        ),
    )
    parser.add_argument("--front-checkpoint", type=Path, required=True)
    parser.add_argument("--back-checkpoint", type=Path, required=True)
    parser.add_argument("--left-checkpoint", type=Path, required=True)
    parser.add_argument("--right-checkpoint", type=Path, required=True)
    parser.add_argument("--front-threshold", type=float, default=0.65)
    parser.add_argument("--back-threshold", type=float, default=0.73)
    parser.add_argument("--left-threshold", type=float, default=0.62)
    parser.add_argument("--right-threshold", type=float, default=0.54)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Repeat inference over the test set this many times for more "
            "stable latency measurements. Accuracy is reported once."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
    )
    return parser.parse_args()


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    values = np.asarray(values_ms, dtype=np.float64)

    return {
        "count": int(values.size),
        "total_ms": float(values.sum()),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self.process.cpu_percent(None)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                memory = self.process.memory_info()
                self.samples.append(
                    {
                        "rss_mb": memory.rss / (1024**2),
                        "vms_mb": memory.vms / (1024**2),
                        "cpu_percent": self.process.cpu_percent(None),
                        "threads": float(self.process.num_threads()),
                    }
                )
            except psutil.Error:
                pass

            self.stop_event.wait(self.interval_seconds)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}

        def get(key: str) -> np.ndarray:
            return np.asarray(
                [sample[key] for sample in self.samples],
                dtype=np.float64,
            )

        rss = get("rss_mb")
        vms = get("vms_mb")
        cpu = get("cpu_percent")
        threads = get("threads")
        logical_cpus = psutil.cpu_count(logical=True) or 1

        return {
            "peak_rss_mb": float(rss.max()),
            "mean_rss_mb": float(rss.mean()),
            "peak_vms_mb": float(vms.max()),
            "peak_cpu_percent_process_scale": float(cpu.max()),
            "mean_cpu_percent_process_scale": float(cpu.mean()),
            "peak_cpu_percent_of_machine": float(cpu.max() / logical_cpus),
            "mean_cpu_percent_of_machine": float(cpu.mean() / logical_cpus),
            "peak_process_threads": int(threads.max()),
            "logical_cpu_count": logical_cpus,
        }


def discover_samples(
    dataset_root: Path,
    view: str,
) -> list[tuple[Path, int]]:
    view_root = dataset_root / "test" / view
    samples: list[tuple[Path, int]] = []

    for label, class_name in enumerate(CLASS_NAMES):
        folder = view_root / class_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected test folder not found: {folder}"
            )

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise RuntimeError(f"No test images found for view: {view}")

    return samples


def clean_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state dictionary."
        )

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        new_key = key

        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def build_model(checkpoint: Path) -> tuple[nn.Module, float]:
    start = time.perf_counter()

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        2,
    )

    raw_state = torch.load(checkpoint, map_location="cpu")
    state_dict = clean_state_dict(raw_state)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    elapsed = time.perf_counter() - start
    return model, elapsed


def warm_up(
    model: nn.Module,
    samples: list[tuple[Path, int]],
    runs: int,
) -> float:
    if runs <= 0:
        return 0.0

    start = time.perf_counter()

    with torch.inference_mode():
        for index in range(runs):
            path, _ = samples[index % len(samples)]
            image = Image.open(path).convert("RGB")
            tensor = INFERENCE_TRANSFORM(image).unsqueeze(0)
            _ = model(tensor)

    return time.perf_counter() - start


def benchmark_view(
    view: str,
    checkpoint: Path,
    threshold: float,
    samples: list[tuple[Path, int]],
    warmup_runs: int,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, model_load_seconds = build_model(checkpoint)
    warmup_seconds = warm_up(model, samples, warmup_runs)

    labels: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, Any]] = []

    image_load_ms: list[float] = []
    preprocessing_ms: list[float] = []
    inference_ms: list[float] = []
    postprocessing_ms: list[float] = []
    end_to_end_ms: list[float] = []

    sampler = ResourceSampler()
    sampler.start()

    benchmark_start = time.perf_counter()

    with torch.inference_mode():
        for repetition in range(repeat):
            for path, true_label in tqdm(
                samples,
                desc=f"{view} repeat {repetition + 1}/{repeat}",
                unit="image",
            ):
                total_start = time.perf_counter()

                load_start = time.perf_counter()
                image = Image.open(path).convert("RGB")
                load_elapsed = (
                    time.perf_counter() - load_start
                ) * 1000.0

                preprocessing_start = time.perf_counter()
                tensor = INFERENCE_TRANSFORM(image).unsqueeze(0)
                preprocessing_elapsed = (
                    time.perf_counter() - preprocessing_start
                ) * 1000.0

                inference_start = time.perf_counter()
                logits = model(tensor)
                inference_elapsed = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                postprocessing_start = time.perf_counter()
                incomplete_probability = float(
                    torch.softmax(logits, dim=1)[0, 1].item()
                )
                predicted_label = int(
                    incomplete_probability >= threshold
                )
                postprocessing_elapsed = (
                    time.perf_counter() - postprocessing_start
                ) * 1000.0

                total_elapsed = (
                    time.perf_counter() - total_start
                ) * 1000.0

                image_load_ms.append(load_elapsed)
                preprocessing_ms.append(preprocessing_elapsed)
                inference_ms.append(inference_elapsed)
                postprocessing_ms.append(postprocessing_elapsed)
                end_to_end_ms.append(total_elapsed)

                # Accuracy is calculated on the first pass only.
                if repetition == 0:
                    labels.append(true_label)
                    predictions.append(predicted_label)

                rows.append(
                    {
                        "view": view,
                        "path": str(path),
                        "repeat": repetition + 1,
                        "true_label": CLASS_NAMES[true_label],
                        "predicted_label": CLASS_NAMES[predicted_label],
                        "correct": predicted_label == true_label,
                        "incomplete_probability": incomplete_probability,
                        "threshold": threshold,
                        "image_load_ms": load_elapsed,
                        "preprocessing_ms": preprocessing_elapsed,
                        "inference_ms": inference_elapsed,
                        "postprocessing_ms": postprocessing_elapsed,
                        "end_to_end_ms": total_elapsed,
                    }
                )

    benchmark_seconds = time.perf_counter() - benchmark_start
    sampler.stop()

    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)

    accuracy = float(
        accuracy_score(labels_array, predictions_array)
    )
    balanced_accuracy = float(
        balanced_accuracy_score(labels_array, predictions_array)
    )
    cm = confusion_matrix(
        labels_array,
        predictions_array,
        labels=[0, 1],
    )

    report = classification_report(
        labels_array,
        predictions_array,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    total_processed = len(samples) * repeat
    throughput = (
        total_processed / benchmark_seconds
        if benchmark_seconds > 0
        else 0.0
    )

    metrics = {
        "view": view,
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "test_images": len(samples),
        "repeat": repeat,
        "total_inferences": total_processed,
        "model_loading_seconds": model_load_seconds,
        "warmup": {
            "runs": warmup_runs,
            "seconds": warmup_seconds,
        },
        "quality": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "complete_recall": float(
                report["complete"]["recall"]
            ),
            "incomplete_recall": float(
                report["incomplete"]["recall"]
            ),
            "complete_precision": float(
                report["complete"]["precision"]
            ),
            "incomplete_precision": float(
                report["incomplete"]["precision"]
            ),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "timing": {
            "image_loading": latency_summary(image_load_ms),
            "preprocessing": latency_summary(preprocessing_ms),
            "model_inference": latency_summary(inference_ms),
            "postprocessing": latency_summary(postprocessing_ms),
            "end_to_end": latency_summary(end_to_end_ms),
            "benchmark_seconds": benchmark_seconds,
            "throughput_images_per_second": throughput,
        },
        "resources": sampler.summary(),
    }

    del model
    return metrics, rows


def main() -> None:
    args = parse_args()

    if args.clear_output and args.output.exists():
        import shutil
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    try:
        import cv2
        cv2.setNumThreads(args.threads)
    except Exception:
        pass

    checkpoints = {
        "front": args.front_checkpoint,
        "back": args.back_checkpoint,
        "left": args.left_checkpoint,
        "right": args.right_checkpoint,
    }

    thresholds = {
        "front": args.front_threshold,
        "back": args.back_threshold,
        "left": args.left_threshold,
        "right": args.right_threshold,
    }

    for checkpoint in checkpoints.values():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    all_metrics: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    total_suite_start = time.perf_counter()

    for view in VIEWS:
        print("\n" + "=" * 76)
        print(f"BENCHMARKING {view.upper()} COMPLETENESS MODEL")
        print("=" * 76)

        samples = discover_samples(args.dataset, view)

        metrics, rows = benchmark_view(
            view=view,
            checkpoint=checkpoints[view],
            threshold=thresholds[view],
            samples=samples,
            warmup_runs=args.warmup_runs,
            repeat=args.repeat,
        )

        all_metrics[view] = metrics
        all_rows.extend(rows)

        quality = metrics["quality"]
        timing = metrics["timing"]
        resources = metrics["resources"]

        print(f"\n{view.upper()} RESULTS")
        print(
            f"Accuracy:             "
            f"{quality['accuracy']:.4%}"
        )
        print(
            f"Balanced accuracy:    "
            f"{quality['balanced_accuracy']:.4%}"
        )
        print(
            f"Complete recall:      "
            f"{quality['complete_recall']:.4%}"
        )
        print(
            f"Incomplete recall:    "
            f"{quality['incomplete_recall']:.4%}"
        )
        print(
            f"Confusion matrix:     "
            f"{quality['confusion_matrix']}"
        )
        print(
            f"Model load time:      "
            f"{metrics['model_loading_seconds']:.4f} s"
        )
        print(
            f"Inference mean:       "
            f"{timing['model_inference']['mean_ms']:.3f} ms"
        )
        print(
            f"Inference p95:        "
            f"{timing['model_inference']['p95_ms']:.3f} ms"
        )
        print(
            f"End-to-end mean:      "
            f"{timing['end_to_end']['mean_ms']:.3f} ms"
        )
        print(
            f"End-to-end p95:       "
            f"{timing['end_to_end']['p95_ms']:.3f} ms"
        )
        print(
            f"Throughput:           "
            f"{timing['throughput_images_per_second']:.2f} img/s"
        )

        if resources:
            print(
                f"Peak RSS:             "
                f"{resources['peak_rss_mb']:.1f} MB"
            )
            print(
                f"Peak machine CPU:     "
                f"{resources['peak_cpu_percent_of_machine']:.1f}%"
            )

    suite_seconds = time.perf_counter() - total_suite_start

    aggregate_inference = []
    aggregate_end_to_end = []
    total_images = 0

    for view, metrics in all_metrics.items():
        total_images += metrics["total_inferences"]

    for row in all_rows:
        aggregate_inference.append(row["inference_ms"])
        aggregate_end_to_end.append(row["end_to_end_ms"])

    summary = {
        "configuration": {
            "dataset": str(args.dataset),
            "device": "cpu",
            "threads": args.threads,
            "views": VIEWS,
            "thresholds": thresholds,
            "checkpoints": {
                view: str(path)
                for view, path in checkpoints.items()
            },
            "warmup_runs_per_view": args.warmup_runs,
            "repeat": args.repeat,
        },
        "per_view": all_metrics,
        "aggregate": {
            "total_inferences": total_images,
            "suite_seconds": suite_seconds,
            "model_inference": latency_summary(
                aggregate_inference
            ),
            "end_to_end": latency_summary(
                aggregate_end_to_end
            ),
            "throughput_images_per_second": (
                total_images / suite_seconds
                if suite_seconds > 0
                else 0.0
            ),
            "sum_model_loading_seconds": sum(
                metrics["model_loading_seconds"]
                for metrics in all_metrics.values()
            ),
        },
    }

    json_path = args.output / "completeness_cpu_benchmark.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if args.save_predictions:
        csv_path = args.output / "completeness_cpu_predictions.csv"

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(all_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        csv_path = None

    aggregate = summary["aggregate"]

    print("\n" + "=" * 76)
    print("ALL COMPLETENESS MODELS — CPU SUMMARY")
    print("=" * 76)
    print(
        f"Aggregate model inference: "
        f"mean={aggregate['model_inference']['mean_ms']:.3f} ms | "
        f"p50={aggregate['model_inference']['p50_ms']:.3f} ms | "
        f"p95={aggregate['model_inference']['p95_ms']:.3f} ms | "
        f"p99={aggregate['model_inference']['p99_ms']:.3f} ms"
    )
    print(
        f"Aggregate end-to-end:      "
        f"mean={aggregate['end_to_end']['mean_ms']:.3f} ms | "
        f"p50={aggregate['end_to_end']['p50_ms']:.3f} ms | "
        f"p95={aggregate['end_to_end']['p95_ms']:.3f} ms | "
        f"p99={aggregate['end_to_end']['p99_ms']:.3f} ms"
    )
    print(
        f"Suite throughput:          "
        f"{aggregate['throughput_images_per_second']:.2f} img/s"
    )
    print(
        f"Sum model loading:         "
        f"{aggregate['sum_model_loading_seconds']:.3f} s"
    )
    print(f"JSON report:               {json_path}")
    if csv_path:
        print(f"Predictions CSV:           {csv_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
