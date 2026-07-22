# Avidea Summer Internship - Vehicle Verification Pipeline

This repository contains a vehicle verification and analysis pipeline built during the Avidea internship. The work so far focuses on preparing image datasets, running vehicle-related computer vision tasks, and evaluating model outputs for tasks such as view classification, color mismatch detection, license plate extraction, match-group validation, and vehicle re-identification.

## What has been implemented

The project currently includes:

- Dataset preparation and validation scripts for raw, cropped, and inference image sets
- View classification workflows for car images
- Vehicle matching and grouping logic
- License plate extraction support
- Vehicle re-identification model assets and related code
- Benchmarking and batch inference utilities for CPU-based experiments

## Repository structure

```text
models/
  vehicle_Re-ID_Kaggle_result/   # Re-ID model weights and training code
  tasks/                          # task-specific logic such as view classification
  utils/                          # shared image and caching helpers

scripts/
  avidea_tasks/
    car_detection/
    color_mismatch_detection/
    license_plate_extraction/
    match_group/
    view_classification/
  benchmarking/
    batch_inference_cpu.py
    benchmark_cpu.py
  data_prep/
    prepare_raw_dataset.py
    prepare_cropped_dataset.py
    prepare_inference_set.py
    prepare_test_dataset.py
    check_data_leakage.py
    verify_Overlap.py
    webp_to_jpeg.py

prompt.py                         # prompt templates used for validation tasks
requirements.txt
```

## Environment setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project relies on PyTorch, torchvision-compatible tooling, Pillow, and other common ML dependencies listed in `requirements.txt`.

## Data preparation workflow

Several scripts are available to prepare and validate datasets before running inference:

```bash
python scripts/data_prep/prepare_raw_dataset.py
python scripts/data_prep/prepare_cropped_dataset.py
python scripts/data_prep/prepare_inference_set.py
python scripts/data_prep/prepare_test_dataset.py
```

Additional checks are available for leakage and overlap analysis:

```bash
python scripts/data_prep/check_data_leakage.py
python scripts/data_prep/verify_Overlap.py
```

## Running the main tasks

The task-specific pipelines are organized under `scripts/avidea_tasks/`.

### View classification

```bash
python scripts/avidea_tasks/view_classification/classify_one_folder_with_output.py
```

### Cropped view classification

```bash
python scripts/avidea_tasks/view_classification/crop_and_classify_one_folder.py
```

Other subtask folders contain the supporting logic for matching, plate extraction, and color-related validation.

## Model assets

Model weights and checkpoints are stored under `models/` and are used by the task scripts. Some of the larger model files are intentionally excluded from version control through the repository ignore rules.

## Notes

This README is intended to reflect the current state of the repository and the workflow being developed for the internship project. As more scripts and models are added, this document can be expanded with the final training, evaluation, and deployment steps.
