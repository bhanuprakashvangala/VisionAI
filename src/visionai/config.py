"""Configuration management using environment variables."""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def get_required_env(key: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}. "
            f"Set it in your .env file or environment. See .env.example for reference."
        )
    return value


def get_hf_token() -> str:
    return get_required_env("HF_TOKEN")


def get_serp_api_key() -> str:
    return get_required_env("SERP_API_KEY")


# Model configuration
BLIP_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

# Output directory
OUTPUT_DIR = Path(os.environ.get("VISIONAI_OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
