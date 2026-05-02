"""Main pipeline orchestrating the VisionAI workflow."""

import argparse
import logging
import sys
from pathlib import Path

from .config import OUTPUT_DIR
from .models import model_manager
from .input_handler import get_user_input
from .location import get_location
from .image_processing import process_image
from .realtime_info import retrieve_real_time_info
from .audio import text_to_speech, evaluate_audio_clarity

logger = logging.getLogger(__name__)

REALTIME_KEYWORDS = {"weather", "hazard", "location", "danger", "emergency", "traffic"}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_ai_response(question: str, scene_description: str) -> str:
    """Generate an AI response to a user question about an image scene."""
    context = f"User Question: {question}\n\nImage Context:\n{scene_description}"
    response = model_manager.generate_llama_response(context)

    # Save response
    response_path = OUTPUT_DIR / "user_response.txt"
    response_path.write_text(response, encoding="utf-8")

    print("\n**LLaMA-2 Response:**\n")
    print(response)
    return response


def needs_realtime_info(question: str) -> bool:
    """Check if the user's question requires real-time data lookup."""
    words = set(question.lower().split())
    return bool(words & REALTIME_KEYWORDS)


def run(image_path: str = "input.jpg", verbose: bool = False) -> None:
    """Execute the full VisionAI pipeline.

    Args:
        image_path: Path to the input image.
        verbose: Enable debug logging.
    """
    setup_logging(verbose)
    logger.info("Starting VisionAI pipeline")
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Step 1: Get user location
    latitude, longitude = get_location()

    # Step 2: Process image
    scene_description = process_image(image_path)
    if scene_description is None:
        logger.error("Image processing failed. Exiting.")
        print("\nImage processing failed. Exiting.")
        sys.exit(1)

    # Step 3: Get user question
    user_question = get_user_input()
    if not user_question:
        logger.warning("No user input received. Exiting.")
        print("\nNo input received. Exiting.")
        sys.exit(1)

    # Step 4: Generate response
    if needs_realtime_info(user_question):
        response = retrieve_real_time_info(user_question, latitude, longitude)
    else:
        response = generate_ai_response(user_question, scene_description)

    # Step 5: Convert to audio
    audio_path = text_to_speech(response)

    # Step 6: Evaluate audio quality
    evaluate_audio_clarity(audio_path)

    print("\nAudio response saved. Processing complete!")
    logger.info("Pipeline completed successfully")


def main() -> None:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="VisionAI - AI-powered assistance for visually impaired individuals"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="input.jpg",
        help="Path to the input image (default: input.jpg)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()
    run(image_path=args.image, verbose=args.verbose)


if __name__ == "__main__":
    main()
