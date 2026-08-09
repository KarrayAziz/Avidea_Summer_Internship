#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import threading
import time
from pathlib import Path
from typing import Any

# Set these before importing NumPy/PyTorch/OpenCV in each process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import psutil


CLASS_NAMES = ["back", "front", "left", "right"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_WORKER_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark four vehicle images in parallel using four persistent "
            "CPU worker processes. Each worker owns one YOLO model and one "
            "ResNet-18 model and uses one CPU inference thread."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--resnet-weights", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--groups", type=int, default=25)
    parser.add_argument("--warmup-groups", type=int, default=2)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--min-area-percent", type=float, default=8.0)
    parser.add_argument(
        "--image-selection",
        choices=["one-per-view", "all"],
        default="one-per-view",
        help=(
            "one-per-view repeatedly benchmarks one image from each of "
            "back/front/left/right. all groups all discovered images by four."
        ),
    )
    return parser.parse_args()


def discover_images(root: Path, mode: str) -> list[Path]:
    if mode == "one-per-view":
        selected: list[Path] = []
        for view in CLASS_NAMES:
            folder = root / view
            candidates = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
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
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            )
        )
    if len(images) < 4:
        raise RuntimeError("At least four images are required.")
    return images


def worker_initializer(
    resnet_weights: str,
    yolo_weights: str,
    threads_per_worker: int,
    yolo_conf: float,
    min_area_percent: float,
) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from ultralytics import YOLO

    torch.set_num_threads(threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
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

    yolo = YOLO(yolo_weights)

    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    state = torch.load(resnet_weights, map_location="cpu")
    resnet.load_state_dict(state)
    resnet.eval()

    _WORKER_STATE.update(
        {
            "torch": torch,
            "yolo": yolo,
            "resnet": resnet,
            "transform": transform,
            "yolo_conf": yolo_conf,
            "min_area_percent": min_area_percent,
        }
    )


def process_one_image(image_path_string: str) -> dict[str, Any]:
    from PIL import Image

    torch = _WORKER_STATE["torch"]
    yolo = _WORKER_STATE["yolo"]
    resnet = _WORKER_STATE["resnet"]
    transform = _WORKER_STATE["transform"]

    image_path = Path(image_path_string)
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    load_ms = (time.perf_counter() - load_start) * 1000.0

    width, height = image.size
    total_area = float(width * height)

    yolo_start = time.perf_counter()
    results = yolo.predict(
        source=str(image_path),
        conf=_WORKER_STATE["yolo_conf"],
        classes=[2, 7],  # car and truck only
        device="cpu",
        verbose=False,
    )
    yolo_ms = (time.perf_counter() - yolo_start) * 1000.0

    used_crop = False
    fallback_reason = "no_detection"
    vehicle_area_percent = 0.0

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.detach().cpu()
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        largest_idx = int(torch.argmax(areas).item())
        largest_area = float(areas[largest_idx].item())
        vehicle_area_percent = largest_area / total_area * 100.0

        if vehicle_area_percent >= _WORKER_STATE["min_area_percent"]:
            x1, y1, x2, y2 = map(int, boxes[largest_idx].tolist())
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            image = image.crop((x1, y1, x2, y2))
            used_crop = True
            fallback_reason = ""
        else:
            fallback_reason = "below_area_threshold"

    preprocessing_start = time.perf_counter()
    tensor = transform(image).unsqueeze(0)
    preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000.0

    inference_start = time.perf_counter()
    with torch.inference_mode():
        logits = resnet(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    predicted_idx = int(torch.argmax(probabilities).item())
    total_ms = (time.perf_counter() - total_start) * 1000.0

    process = psutil.Process(os.getpid())

    return {
        "path": str(image_path),
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": float(probabilities[predicted_idx].item()),
        "used_crop": used_crop,
        "fallback_reason": fallback_reason,
        "vehicle_area_percent": vehicle_area_percent,
        "load_ms": load_ms,
        "yolo_ms": yolo_ms,
        "preprocessing_ms": preprocessing_ms,
        "resnet_ms": inference_ms,
        "worker_total_ms": total_ms,
        "worker_pid": os.getpid(),
        "worker_rss_mb": process.memory_info().rss / (1024 ** 2),
    }


class ResourceSampler:
    def __init__(self, parent_pid: int, interval_seconds: float = 0.05):
        self.parent_pid = parent_pid
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        parent = psutil.Process(self.parent_pid)
        parent.cpu_percent(None)
        for child in parent.children(recursive=True):
            try:
                child.cpu_percent(None)
            except psutil.Error:
                pass
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                parent = psutil.Process(self.parent_pid)
                processes = [parent] + parent.children(recursive=True)

                rss_bytes = 0
                cpu_percent = 0.0
                thread_count = 0

                for process in processes:
                    try:
                        rss_bytes += process.memory_info().rss
                        cpu_percent += process.cpu_percent(None)
                        thread_count += process.num_threads()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                self.samples.append(
                    {
                        "rss_mb": rss_bytes / (1024 ** 2),
                        "cpu_percent_sum": cpu_percent,
                        "thread_count": float(thread_count),
                    }
                )
            except psutil.Error:
                pass

            self.stop_event.wait(self.interval_seconds)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}

        rss = np.array([s["rss_mb"] for s in self.samples], dtype=float)
        cpu = np.array(
            [s["cpu_percent_sum"] for s in self.samples],
            dtype=float,
        )
        threads = np.array(
            [s["thread_count"] for s in self.samples],
            dtype=float,
        )

        logical_cpus = psutil.cpu_count(logical=True) or 1

        return {
            "peak_total_rss_mb": float(rss.max()),
            "mean_total_rss_mb": float(rss.mean()),
            "peak_cpu_percent_sum": float(cpu.max()),
            "mean_cpu_percent_sum": float(cpu.mean()),
            "peak_cpu_percent_of_machine": float(
                cpu.max() / logical_cpus
            ),
            "mean_cpu_percent_of_machine": float(
                cpu.mean() / logical_cpus
            ),
            "peak_process_thread_count": int(threads.max()),
            "mean_process_thread_count": float(threads.mean()),
            "logical_cpu_count": logical_cpus,
        }


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
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


def make_groups(images: list[Path], groups: int, mode: str) -> list[list[Path]]:
    if mode == "one-per-view":
        return [images[:] for _ in range(groups)]

    result: list[list[Path]] = []
    cursor = 0
    for _ in range(groups):
        group: list[Path] = []
        for _ in range(4):
            group.append(images[cursor % len(images)])
            cursor += 1
        result.append(group)
    return result


def main() -> None:
    args = parse_args()

    if args.workers != 4:
        print(
            "Warning: this benchmark is designed for four simultaneous images; "
            f"workers={args.workers}."
        )

    for path in [
        args.input_dir,
        args.resnet_weights,
        args.yolo_weights,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.input_dir, args.image_selection)
    groups = make_groups(images, args.groups, args.image_selection)

    context = mp.get_context("spawn")

    pool_start = time.perf_counter()
    pool = context.Pool(
        processes=args.workers,
        initializer=worker_initializer,
        initargs=(
            str(args.resnet_weights),
            str(args.yolo_weights),
            args.threads_per_worker,
            args.yolo_conf,
            args.min_area_percent,
        ),
    )

    # Force all workers to initialize before timing actual groups.
    initialization_paths = [
        str(images[index % len(images)])
        for index in range(args.workers)
    ]
    _ = pool.map(process_one_image, initialization_paths, chunksize=1)
    pool_initialization_seconds = time.perf_counter() - pool_start

    print("=" * 76)
    print("FOUR-IMAGE PARALLEL CPU BENCHMARK")
    print("=" * 76)
    print(f"Workers:             {args.workers}")
    print(f"Threads per worker:  {args.threads_per_worker}")
    print(f"Groups measured:     {args.groups}")
    print(f"Warm-up groups:      {args.warmup_groups}")
    print(f"Pool/model startup:  {pool_initialization_seconds:.3f} s")
    print()

    for _ in range(args.warmup_groups):
        pool.map(
            process_one_image,
            [str(path) for path in groups[0]],
            chunksize=1,
        )

    sampler = ResourceSampler(os.getpid())
    sampler.start()

    group_latencies_ms: list[float] = []
    individual_results: list[dict[str, Any]] = []

    benchmark_start = time.perf_counter()

    for group_index, group in enumerate(groups, start=1):
        start = time.perf_counter()

        results = pool.map(
            process_one_image,
            [str(path) for path in group],
            chunksize=1,
        )

        group_latency_ms = (time.perf_counter() - start) * 1000.0
        group_latencies_ms.append(group_latency_ms)

        for result in results:
            result["group_index"] = group_index
            result["group_latency_ms"] = group_latency_ms
            individual_results.append(result)

        print(
            f"Group {group_index:03d}: "
            f"{group_latency_ms:.2f} ms for 4 images"
        )

    total_benchmark_seconds = time.perf_counter() - benchmark_start

    sampler.stop()
    resource_summary = sampler.summary()

    pool.close()
    pool.join()

    four_image_latency = latency_summary(group_latencies_ms)
    individual_worker_latency = latency_summary(
        [row["worker_total_ms"] for row in individual_results]
    )

    total_images = len(groups) * 4
    throughput = total_images / total_benchmark_seconds

    sequential_reference_ms = (
        sum(row["worker_total_ms"] for row in individual_results)
        / len(groups)
    )
    measured_speedup = (
        sequential_reference_ms / four_image_latency["mean_ms"]
        if four_image_latency["mean_ms"] > 0
        else 0.0
    )

    worker_rss_by_pid: dict[str, float] = {}
    for row in individual_results:
        pid = str(row["worker_pid"])
        worker_rss_by_pid[pid] = max(
            worker_rss_by_pid.get(pid, 0.0),
            row["worker_rss_mb"],
        )

    report = {
        "configuration": {
            "input_dir": str(args.input_dir),
            "workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "groups": args.groups,
            "warmup_groups": args.warmup_groups,
            "image_selection": args.image_selection,
            "yolo_classes": [2, 7],
            "minimum_area_percent": args.min_area_percent,
        },
        "startup": {
            "pool_and_model_initialization_seconds":
                pool_initialization_seconds,
        },
        "latency": {
            "four_image_group": four_image_latency,
            "individual_worker": individual_worker_latency,
            "estimated_sequential_four_image_mean_ms":
                sequential_reference_ms,
            "measured_parallel_speedup_vs_worker_time_sum":
                measured_speedup,
        },
        "throughput": {
            "total_images": total_images,
            "total_benchmark_seconds": total_benchmark_seconds,
            "images_per_second": throughput,
            "four_image_groups_per_second":
                len(groups) / total_benchmark_seconds,
        },
        "resources": {
            **resource_summary,
            "peak_rss_mb_per_worker_pid": worker_rss_by_pid,
            "sum_peak_worker_rss_mb": sum(worker_rss_by_pid.values()),
        },
        "predictions": individual_results,
    }

    report_path = args.output / "parallel_four_image_benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(
        "Four-image latency: "
        f"mean={four_image_latency['mean_ms']:.2f} ms | "
        f"p50={four_image_latency['p50_ms']:.2f} ms | "
        f"p95={four_image_latency['p95_ms']:.2f} ms | "
        f"p99={four_image_latency['p99_ms']:.2f} ms"
    )
    print(
        "Estimated sequential time for the same four worker tasks: "
        f"{sequential_reference_ms:.2f} ms"
    )
    print(f"Measured speedup:    {measured_speedup:.2f}x")
    print(f"Throughput:          {throughput:.2f} images/s")
    print(f"Pool/model startup:  {pool_initialization_seconds:.2f} s")

    if resource_summary:
        print("\nResource usage:")
        print(
            f"  Peak total RSS:           "
            f"{resource_summary['peak_total_rss_mb']:.1f} MB"
        )
        print(
            f"  Mean total RSS:           "
            f"{resource_summary['mean_total_rss_mb']:.1f} MB"
        )
        print(
            f"  Sum peak worker RSS:      "
            f"{sum(worker_rss_by_pid.values()):.1f} MB"
        )
        print(
            f"  Peak CPU usage:           "
            f"{resource_summary['peak_cpu_percent_sum']:.1f}% "
            f"(psutil summed process scale)"
        )
        print(
            f"  Peak machine CPU share:   "
            f"{resource_summary['peak_cpu_percent_of_machine']:.1f}%"
        )
        print(
            f"  Peak process threads:     "
            f"{resource_summary['peak_process_thread_count']}"
        )

    print(f"\nReport: {report_path}")
    print("=" * 76)


if __name__ == "__main__":
    mp.freeze_support()
    main()
