"""Model loading and management with lazy initialization."""

import logging
from typing import Optional

import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from .config import BLIP_MODEL_NAME, LLAMA_MODEL_NAME, TTS_MODEL_NAME, get_hf_token

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    """Manages lazy loading and lifecycle of AI models."""

    def __init__(self) -> None:
        self._blip_processor: Optional[Blip2Processor] = None
        self._blip_model: Optional[Blip2ForConditionalGeneration] = None
        self._llama_tokenizer: Optional[AutoTokenizer] = None
        self._llama_model: Optional[AutoModelForCausalLM] = None
        self._tts_model = None

    @property
    def blip_processor(self) -> Blip2Processor:
        if self._blip_processor is None:
            logger.info("Loading BLIP-2 processor: %s", BLIP_MODEL_NAME)
            self._blip_processor = Blip2Processor.from_pretrained(
                BLIP_MODEL_NAME, token=get_hf_token()
            )
        return self._blip_processor

    @property
    def blip_model(self) -> Blip2ForConditionalGeneration:
        if self._blip_model is None:
            logger.info("Loading BLIP-2 model: %s", BLIP_MODEL_NAME)
            self._blip_model = Blip2ForConditionalGeneration.from_pretrained(
                BLIP_MODEL_NAME,
                torch_dtype=torch.float16,
                token=get_hf_token(),
            ).to(DEVICE)
        return self._blip_model

    @property
    def llama_tokenizer(self) -> AutoTokenizer:
        if self._llama_tokenizer is None:
            logger.info("Loading LLaMA-2 tokenizer: %s", LLAMA_MODEL_NAME)
            self._llama_tokenizer = AutoTokenizer.from_pretrained(
                LLAMA_MODEL_NAME, token=get_hf_token()
            )
        return self._llama_tokenizer

    @property
    def llama_model(self) -> AutoModelForCausalLM:
        if self._llama_model is None:
            logger.info("Loading LLaMA-2 model: %s", LLAMA_MODEL_NAME)
            self._llama_model = AutoModelForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                torch_dtype=torch.float16,
                token=get_hf_token(),
            ).to(DEVICE)
        return self._llama_model

    @property
    def tts_model(self):
        if self._tts_model is None:
            from TTS.api import TTS

            logger.info("Loading TTS model: %s", TTS_MODEL_NAME)
            self._tts_model = TTS(
                model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False
            )
        return self._tts_model

    def generate_llama_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate text using LLaMA-2 from a prompt."""
        inputs = self.llama_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()

        output = self.llama_model.generate(**inputs, max_length=max_length)
        return self.llama_tokenizer.decode(output[0], skip_special_tokens=True)

    def unload_all(self) -> None:
        """Release all models from memory."""
        self._blip_processor = None
        self._blip_model = None
        self._llama_tokenizer = None
        self._llama_model = None
        self._tts_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")


# Singleton instance
model_manager = ModelManager()
