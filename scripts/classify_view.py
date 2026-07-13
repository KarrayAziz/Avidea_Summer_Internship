import argparse
import json
import logging
from pathlib import Path

from models.embeddings.siglip2 import SigLIP2EmbeddingModel
from models.tasks.view_classifier import ViewClassifier
from models.utils.image import load_image


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one vehicle image view with SigLIP2 on CPU.")
    parser.add_argument("image", type=Path, help="Path to a jpg, jpeg, png, or webp image.")
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--prompt-set", default="prompt_set_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prompt-score-aggregation",
        choices=("mean", "max"),
        default="mean",
        help="How to combine multiple prompts for the same label.",
    )
    parser.add_argument(
        "--probability-mode",
        choices=("sigmoid_normalized", "softmax"),
        default="sigmoid_normalized",
        help="Use sigmoid_normalized for SigLIP-style logits, or softmax for CLIP-style scores.",
    )
    parser.add_argument(
        "--null-margin",
        type=float,
        default=0.5,
        help="Raw-score margin null must beat the best vehicle view by before predicting null.",
    )
    parser.add_argument("--hide-raw-scores", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading image: %s", args.image)
    image = load_image(args.image)

    classifier = ViewClassifier(
        embedding_model=SigLIP2EmbeddingModel(model_id=args.model_id, device=args.device),
        prompt_set=args.prompt_set,
        probability_mode=args.probability_mode,
        prompt_score_aggregation=args.prompt_score_aggregation,
        null_margin=args.null_margin,
    )
    result = classifier.classify(image)

    print(json.dumps(result.to_dict(include_raw_scores=not args.hide_raw_scores), indent=2))


if __name__ == "__main__":
    main()
