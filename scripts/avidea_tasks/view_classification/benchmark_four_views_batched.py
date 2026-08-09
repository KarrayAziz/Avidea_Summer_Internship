#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# Apply thread limits before importing native ML libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import psutil
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


CLASS_NAMES = ["back", "front", "left", "right"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_CLASSES = [2, 7]  # car, truck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark four vehicle images as one YOLO batch followed by one "
            "ResNet-18 batch. Measures four-image latency and resource use."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--resnet-weights", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--warmup-groups", type=int, default=3)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--min-area-percent", type=float, default=8.0)
    parser.add_argument(
        "--image-selection",
        choices=["one-per-view", "all"],
        default="one-per-view",
    )
    return parser.parse_args()


def discover_images(root: Path, mode: str) -> list[Path]:
    if mode == "one-per-view":
        selected: list[Path] = []
        for view in CLASS_NAMES:
            folder = root / view
            candidates = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            )
            if not candidates:
                raise RuntimeError(f"No image found for view: {view}")
            selected.append(candidates[0])
        return selected

    images: list[Path] = []
    for view in CLASS_NAMES:
        folder = root / view
        if not folder.exists():
            continue
        images.extend(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            )
        )

    if len(images) < 4:
        raise RuntimeError("At least four images are required.")

    return images


def build_groups(
    images: list[Path],
    group_count: int,
    mode: str,
) -> list[list[Path]]:
    if mode == "one-per-view":
        return [images[:] for _ in range(group_count)]

    groups: list[list[Path]] = []
    cursor = 0

    for _ in range(group_count):
        group: list[Path] = []
        for _ in range(4):
            group.append(images[cursor % len(images)])
            cursor += 1
        groups.append(group)

    return groups


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "min_ms": float(array.min()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.process.cpu_percent(None)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
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

        def values(key: str) -> np.ndarray:
            return np.asarray(
                [sample[key] for sample in self.samples],
                dtype=np.float64,
            )

        rss = values("rss_mb")
        vms = values("vms_mb")
        cpu = values("cpu_percent")
        threads = values("threads")
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


def load_models(
    resnet_weights: Path,
    yolo_weights: Path,
) -> tuple[YOLO, nn.Module, float, float]:
    yolo_start = time.perf_counter()
    yolo = YOLO(str(yolo_weights))
    yolo_seconds = time.perf_counter() - yolo_start

    resnet_start = time.perf_counter()
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    state = torch.load(resnet_weights, map_location="cpu")
    resnet.load_state_dict(state)
    resnet.eval()
    resnet_seconds = time.perf_counter() - resnet_start

    return yolo, resnet, yolo_seconds, resnet_seconds


def process_group(
    paths: list[Path],
    yolo: YOLO,
    resnet: nn.Module,
    transform: transforms.Compose,
    yolo_conf: float,
    min_area_percent: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    group_start = time.perf_counter()

    load_start = time.perf_counter()
    raw_images = [Image.open(path).convert("RGB") for path in paths]
    load_ms = (time.perf_counter() - load_start) * 1000.0

    yolo_start = time.perf_counter()
    yolo_results = yolo.predict(
        source=[str(path) for path in paths],
        conf=yolo_conf,
        classes=YOLO_CLASSES,
        device="cpu",
        verbose=False,
        stream=False,
    )
    yolo_ms = (time.perf_counter() - yolo_start) * 1000.0

    crop_start = time.perf_counter()
    model_images: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []

    for image, result in zip(raw_images, yolo_results):
        width, height = image.size
        image_area = float(width * height)

        used_crop = False
        fallback_reason = "no_detection"
        vehicle_area_percent = 0.0
        crop_box = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu()
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights

            largest_idx = int(torch.argmax(areas).item())
            largest_area = float(areas[largest_idx].item())
            vehicle_area_percent = largest_area / image_area * 100.0

            if vehicle_area_percent >= min_area_percent:
                x1, y1, x2, y2 = map(
                    int,
                    boxes[largest_idx].tolist(),
                )
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))

                image = image.crop((x1, y1, x2, y2))
                used_crop = True
                fallback_reason = ""
                crop_box = [x1, y1, x2, y2]
            else:
                fallback_reason = "below_area_threshold"

        model_images.append(image)
        metadata.append(
            {
                "used_crop": used_crop,
                "fallback_reason": fallback_reason,
                "vehicle_area_percent": vehicle_area_percent,
                "crop_box": crop_box,
            }
        )

    crop_ms = (time.perf_counter() - crop_start) * 1000.0

    preprocessing_start = time.perf_counter()
    batch_tensor = torch.stack(
        [transform(image) for image in model_images],
        dim=0,
    )
    preprocessing_ms = (
        time.perf_counter() - preprocessing_start
    ) * 1000.0

    inference_start = time.perf_counter()
    with torch.inference_mode():
        logits = resnet(batch_tensor)
        probabilities = torch.softmax(logits, dim=1)
    resnet_ms = (time.perf_counter() - inference_start) * 1000.0

    outputs: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        predicted_idx = int(
            torch.argmax(probabilities[index]).item()
        )
        outputs.append(
            {
                "path": str(path),
                "prediction": CLASS_NAMES[predicted_idx],
                "confidence": float(
                    probabilities[index, predicted_idx].item()
                ),
                **metadata[index],
            }
        )

    group_ms = (time.perf_counter() - group_start) * 1000.0

    timing = {
        "image_loading_ms": load_ms,
        "yolo_batch_ms": yolo_ms,
        "crop_selection_ms": crop_ms,
        "preprocessing_batch_ms": preprocessing_ms,
        "resnet_batch_ms": resnet_ms,
        "four_image_end_to_end_ms": group_ms,
    }

    return outputs, timing


def main() -> None:
    args = parse_args()

    for path in (
        args.input_dir,
        args.resnet_weights,
        args.yolo_weights,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

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

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    images = discover_images(args.input_dir, args.image_selection)
    groups = build_groups(
        images,
        args.groups,
        args.image_selection,
    )

    yolo, resnet, yolo_load_seconds, resnet_load_seconds = load_models(
        args.resnet_weights,
        args.yolo_weights,
    )

    for _ in range(args.warmup_groups):
        process_group(
            paths=groups[0],
            yolo=yolo,
            resnet=resnet,
            transform=transform,
            yolo_conf=args.yolo_conf,
            min_area_percent=args.min_area_percent,
        )

    sampler = ResourceSampler()
    sampler.start()

    timing_rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []

    benchmark_start = time.perf_counter()

    for group_index, group in enumerate(groups, start=1):
        outputs, timing = process_group(
            paths=group,
            yolo=yolo,
            resnet=resnet,
            transform=transform,
            yolo_conf=args.yolo_conf,
            min_area_percent=args.min_area_percent,
        )

        timing["group_index"] = float(group_index)
        timing_rows.append(timing)

        for output in outputs:
            output["group_index"] = group_index
            predictions.append(output)

        print(
            f"Group {group_index:03d}: "
            f"{timing['four_image_end_to_end_ms']:.2f} ms"
        )

    benchmark_seconds = time.perf_counter() - benchmark_start
    sampler.stop()

    metric_names = [
        "image_loading_ms",
        "yolo_batch_ms",
        "crop_selection_ms",
        "preprocessing_batch_ms",
        "resnet_batch_ms",
        "four_image_end_to_end_ms",
    ]

    timing_summary = {
        name: latency_summary(
            [row[name] for row in timing_rows]
        )
        for name in metric_names
    }

    total_images = len(groups) * 4
    throughput = total_images / benchmark_seconds

    report = {
        "configuration": {
            "input_dir": str(args.input_dir),
            "threads": args.threads,
            "group_size": 4,
            "groups": args.groups,
            "warmup_groups": args.warmup_groups,
            "image_selection": args.image_selection,
            "yolo_classes": YOLO_CLASSES,
            "minimum_area_percent": args.min_area_percent,
            "architecture": (
                "One YOLO batch of four images, then one ResNet batch "
                "of four cropped/raw images."
            ),
        },
        "model_loading": {
            "yolo_seconds": yolo_load_seconds,
            "resnet_seconds": resnet_load_seconds,
            "total_seconds": yolo_load_seconds + resnet_load_seconds,
        },
        "timing": timing_summary,
        "throughput": {
            "total_images": total_images,
            "total_groups": len(groups),
            "benchmark_seconds": benchmark_seconds,
            "images_per_second": throughput,
            "groups_per_second": len(groups) / benchmark_seconds,
        },
        "resources": sampler.summary(),
        "predictions": predictions,
    }

    output_path = args.output / "batched_four_image_benchmark.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    group_stats = timing_summary["four_image_end_to_end_ms"]
    yolo_stats = timing_summary["yolo_batch_ms"]
    resnet_stats = timing_summary["resnet_batch_ms"]
    resources = report["resources"]

    print("\n" + "=" * 76)
    print("BATCHED FOUR-IMAGE RESULTS")
    print("=" * 76)
    print(
        f"Four-image latency: mean={group_stats['mean_ms']:.2f} ms | "
        f"p50={group_stats['p50_ms']:.2f} ms | "
        f"p95={group_stats['p95_ms']:.2f} ms | "
        f"p99={group_stats['p99_ms']:.2f} ms"
    )
    print(
        f"YOLO batch:         mean={yolo_stats['mean_ms']:.2f} ms | "
        f"p95={yolo_stats['p95_ms']:.2f} ms"
    )
    print(
        f"ResNet batch:       mean={resnet_stats['mean_ms']:.2f} ms | "
        f"p95={resnet_stats['p95_ms']:.2f} ms"
    )
    print(f"Throughput:         {throughput:.2f} images/s")
    print(
        f"Model load time:    "
        f"{yolo_load_seconds + resnet_load_seconds:.3f} s"
    )

    if resources:
        print("\nResource usage:")
        print(f"  Peak RSS:          {resources['peak_rss_mb']:.1f} MB")
        print(f"  Mean RSS:          {resources['mean_rss_mb']:.1f} MB")
        print(
            f"  Peak CPU:          "
            f"{resources['peak_cpu_percent_process_scale']:.1f}%"
        )
        print(
            f"  Machine CPU share: "
            f"{resources['peak_cpu_percent_of_machine']:.1f}%"
        )
        print(
            f"  Peak threads:      "
            f"{resources['peak_process_threads']}"
        )

    print(f"\nReport: {output_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
