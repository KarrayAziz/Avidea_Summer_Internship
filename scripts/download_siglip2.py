import argparse
import logging


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache a SigLIP2 model from Hugging Face.")
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from transformers import AutoModel, AutoProcessor

    logger.info("Downloading processor: %s", args.model_id)
    AutoProcessor.from_pretrained(args.model_id)

    logger.info("Downloading model: %s", args.model_id)
    AutoModel.from_pretrained(args.model_id)

    logger.info("SigLIP2 is cached locally: %s", args.model_id)


if __name__ == "__main__":
    main()
