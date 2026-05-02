"""Tests for audio evaluation utilities."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile


def _create_test_wav(path: str, duration: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a test WAV file with a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio)
    return path


class TestEvaluateAudioClarity:
    def test_returns_positive_snr_for_clean_signal(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        _create_test_wav(wav_path)
        with patch("visionai.audio.OUTPUT_DIR", tmp_path):
            from visionai.audio import evaluate_audio_clarity
            snr = evaluate_audio_clarity(wav_path)
            assert snr > 0

    def test_missing_file_raises(self):
        from visionai.audio import evaluate_audio_clarity
        with pytest.raises(FileNotFoundError):
            evaluate_audio_clarity("/nonexistent/audio.wav")

    def test_generates_spectrogram(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        _create_test_wav(wav_path)
        with patch("visionai.audio.OUTPUT_DIR", tmp_path):
            from visionai.audio import evaluate_audio_clarity
            evaluate_audio_clarity(wav_path)
            assert (tmp_path / "spectrogram.png").exists()


class TestSaveSpectrogram:
    def test_creates_spectrogram_file(self, tmp_path):
        with patch("visionai.audio.OUTPUT_DIR", tmp_path):
            from visionai.audio import _save_spectrogram
            audio_data = np.sin(np.linspace(0, 100, 16000)).astype(np.float64)
            result = _save_spectrogram(audio_data, 16000)
            assert result.exists()
