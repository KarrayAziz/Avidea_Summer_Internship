# Vehicle Verification API

FastAPI service for vehicle image validation. The existing Gemini and local
llama.cpp endpoints are preserved, and the project now includes a model-agnostic
local embedding path for benchmarking zero-shot computer vision subtasks.

## Architecture

```text
models/
  embeddings/
    base.py        # EmbeddingModel interface
    siglip2.py     # Google SigLIP 2 implementation
  tasks/
    prompts.py
    view_classifier.py
  benchmark/
    benchmark_view.py
    metrics.py
    visualization.py
  utils/
    cache.py
    image.py
```

Task modules depend only on `EmbeddingModel`. Model implementations do not know
about vehicle tasks, labels, or benchmark folders.

## Installation

```bash
conda activate ml
pip install -r requirements.txt
```

Set the existing Gemini environment variables in `.env`:

```env
GEMINI_API_KEY=...
GEMINI_MODEL=...
```

Optional local classifier settings:

```env
SIGLIP2_MODEL=google/siglip2-base-patch16-224
VIEW_CLASSIFIER_PROMPT_SET=prompt_set_1
VIEW_CLASSIFIER_DEVICE=cuda
```

If `VIEW_CLASSIFIER_DEVICE` is unset, SigLIP2 uses CUDA when available and CPU
otherwise.

For CPU-only SigLIP2 work, cache the model once:

```bash
python -m scripts.download_siglip2
```

Classify one image from the terminal:

```bash
python -m scripts.classify_view /path/to/car.jpg
```

Both scripts default to `google/siglip2-base-patch16-224` and CPU inference.
The classifier prints normalized `scores` plus SigLIP2 raw `raw_scores`; hide
raw scores with `--hide-raw-scores`.

For left/right side views, try the direction-aware prompt ensemble:

```bash
python -m scripts.classify_view /path/to/car.jpg --prompt-set directional
```

The classifier only predicts `null` when its raw score is clearly ahead of the
best vehicle-view score. Tune that margin when needed:

```bash
python -m scripts.classify_view /path/to/car.jpg --null-margin 0.25
```

## Running The API

```bash
uvicorn main:app --reload
```

Existing endpoints remain available:

- `POST /verify-vehicle`
- `POST /test-local-llava-authenticity`

New endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/view-classifier" \
  -F "image=@/path/to/car.jpg"
```

Response:

```json
{
  "prediction": "front",
  "confidence": 0.52,
  "scores": {
    "front": 0.52,
    "rear": 0.12,
    "left": 0.18,
    "right": 0.15,
    "null": 0.03
  }
}
```

Supported image formats are `jpg`, `jpeg`, `png`, and `webp`. Images are
converted to RGB automatically.

## Benchmarking View Classification

Expected dataset layout:

```text
dataset/
  front/
  rear/
  left/
  right/
  null/
```

Run:

```bash
python -m models.benchmark.benchmark_view dataset \
  --output-dir benchmark_outputs/view_classifier \
  --model siglip2 \
  --model-id google/siglip2-base-patch16-224 \
  --prompt-set prompt_set_1
```

The benchmark recursively loads supported images, warms up the model, ignores
the first measured inference for latency, and writes:

- `results.csv`
- `metrics.json`
- `confusion_matrix.png`

`results.csv` columns:

```text
filename,ground_truth,prediction,confidence,score_front,score_rear,score_left,score_right,score_null
```

Metrics include overall accuracy, per-class accuracy, precision, recall,
F1-score, confusion matrix, average latency, median latency, P95 latency, and
images/sec.

## Adding Embedding Models

Create a new file in `models/embeddings/` that implements
`models.embeddings.base.EmbeddingModel`:

```python
class MyEmbeddingModel(EmbeddingModel):
    def load(self): ...
    def encode_image(self, image): ...
    def encode_text(self, texts): ...
    def similarity(self, image_embeddings, text_embeddings): ...
```

Keep model-specific preprocessing, device placement, batching, and embedding
normalization inside the embedding implementation. Do not put task labels or
vehicle-specific logic there.

## Adding Prompt Sets

Add a `PromptSet` in `models/tasks/prompts.py` and register it in
`PROMPT_SETS`. Each view prompt set must provide:

- `front`
- `rear`
- `left`
- `right`
- `null`

Select it with `VIEW_CLASSIFIER_PROMPT_SET` or the benchmark `--prompt-set`
argument.

## Adding Tasks

Create a new task file in `models/tasks/` and depend only on
`EmbeddingModel`. A future task such as `is_real`, `is_car`, `matches_group`,
`completeness`, or `plate OCR` should own its prompts and decision logic in the
task layer, while reusing embedding implementations unchanged.
