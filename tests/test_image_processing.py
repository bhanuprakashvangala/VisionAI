"""Tests for image processing utilities."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


def _create_test_image(path: str, size: tuple = (100, 100)) -> str:
    """Create a simple test image."""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path)
    return path


class TestValidateImagePath:
    def test_valid_image(self, tmp_path):
        img_path = str(tmp_path / "test.jpg")
        _create_test_image(img_path)
        from visionai.image_processing import validate_image_path
        result = validate_image_path(img_path)
        assert result == Path(img_path)

    def test_missing_file_raises(self):
        from visionai.image_processing import validate_image_path
        with pytest.raises(FileNotFoundError):
            validate_image_path("/nonexistent/image.jpg")

    def test_unsupported_format_raises(self, tmp_path):
        txt_path = tmp_path / "file.txt"
        txt_path.write_text("not an image")
        from visionai.image_processing import validate_image_path
        with pytest.raises(ValueError, match="Unsupported image format"):
            validate_image_path(str(txt_path))


class TestImageAnalysis:
    def test_save_edge_detection(self, tmp_path):
        with patch("visionai.image_processing.OUTPUT_DIR", tmp_path):
            from visionai.image_processing import _save_edge_detection
            image_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = _save_edge_detection(image_np)
            assert result.exists()

    def test_save_enhanced_image(self, tmp_path):
        with patch("visionai.image_processing.OUTPUT_DIR", tmp_path):
            from visionai.image_processing import _save_enhanced_image
            image_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = _save_enhanced_image(image_np)
            assert result.exists()

    def test_save_histogram(self, tmp_path):
        with patch("visionai.image_processing.OUTPUT_DIR", tmp_path):
            from visionai.image_processing import _save_histogram
            image_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = _save_histogram(image_np)
            assert result.exists()
