"""Image processing, analysis, and scene understanding."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

BLIP_INPUT_SIZE = (384, 384)


def validate_image_path(image_path: str) -> Path:
    """Validate that the image file exists and is a supported format."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. Supported: {supported}"
        )
    return path


def _save_edge_detection(image_np: np.ndarray) -> Path:
    """Generate and save edge-detected version of the image."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    path = OUTPUT_DIR / "edges_input.png"
    cv2.imwrite(str(path), edges)
    return path


def _save_enhanced_image(image_np: np.ndarray) -> Path:
    """Generate and save enhanced version of the image."""
    enhanced = cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75)
    path = OUTPUT_DIR / "enhanced_input.png"
    cv2.imwrite(str(path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    return path


def _save_histogram(image_np: np.ndarray) -> Path:
    """Generate and save color histogram of the image."""
    fig, ax = plt.subplots()
    for i, color in enumerate(("red", "green", "blue")):
        ax.hist(image_np[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=color)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    path = OUTPUT_DIR / "histogram_input.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def process_image(image_path: str) -> Optional[str]:
    """Analyze an image using BLIP-2 and LLaMA-2 to generate a scene description.

    Args:
        image_path: Path to the input image file.

    Returns:
        AI-generated scene description, or None on failure.
    """
    try:
        path = validate_image_path(image_path)
        image = Image.open(path).convert("RGB")
        image_resized = image.resize(BLIP_INPUT_SIZE)

        # Save processed image
        processed_path = OUTPUT_DIR / "processed_input.png"
        image_resized.save(processed_path)

        # Generate visual analysis outputs
        image_np = np.array(image_resized)
        _save_edge_detection(image_np)
        _save_enhanced_image(image_np)
        _save_histogram(image_np)
        logger.info("Image preprocessing complete.")

        # Generate BLIP-2 caption
        import torch
        from .models import model_manager, DEVICE

        blip_inputs = model_manager.blip_processor(
            image_resized, return_tensors="pt"
        ).to(DEVICE, torch.float16)
        blip_out = model_manager.blip_model.generate(**blip_inputs, max_length=200)
        blip_description = model_manager.blip_processor.decode(
            blip_out[0], skip_special_tokens=True
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BLIP-2 caption: %s", blip_description)

        # Enhance description with LLaMA-2
        prompt = (
            f"Analyze the image based on the following description:\n\n"
            f"BLIP-2 Scene Analysis: {blip_description}\n\n"
            f"Please synthesize a detailed, coherent, and enriched description."
        )
        final_description = model_manager.generate_llama_response(prompt)

        print("\n**Final AI-Generated Image Description:**\n")
        print(final_description)

        # Save description
        desc_path = OUTPUT_DIR / "scene_description.txt"
        desc_path.write_text(final_description, encoding="utf-8")

        return final_description

    except FileNotFoundError as e:
        logger.error("Image file error: %s", e)
        print(f"\nError: {e}")
        return None
    except ValueError as e:
        logger.error("Image validation error: %s", e)
        print(f"\nError: {e}")
        return None
    except Exception as e:
        logger.error("Image processing failed: %s", e, exc_info=True)
        print(f"\nError processing image: {e}")
        return None
