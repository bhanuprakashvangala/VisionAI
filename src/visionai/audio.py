"""Audio generation and quality evaluation."""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def text_to_speech(text: str, output_filename: str = "response_audio.wav") -> Path:
    """Convert text to speech audio file.

    Args:
        text: Text to synthesize.
        output_filename: Name of the output WAV file.

    Returns:
        Path to the generated audio file.
    """
    from .models import model_manager

    audio_path = OUTPUT_DIR / output_filename
    model_manager.tts_model.tts_to_file(text=text, file_path=str(audio_path))
    logger.info("Audio saved to %s", audio_path)
    return audio_path


def evaluate_audio_clarity(audio_path: str | Path) -> float:
    """Evaluate audio clarity using Signal-to-Noise Ratio (SNR).

    Estimates noise from silent segments of the audio rather than
    using synthetic random noise.

    Args:
        audio_path: Path to the WAV audio file.

    Returns:
        SNR value in dB.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sample_rate, audio_data = wavfile.read(audio_path)

    # Convert stereo to mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    audio_data = audio_data.astype(np.float64)

    # Estimate noise from the quietest 10% of the signal (likely silence/noise floor)
    frame_size = max(1, len(audio_data) // 100)
    frames = [
        audio_data[i : i + frame_size]
        for i in range(0, len(audio_data) - frame_size, frame_size)
    ]
    frame_powers = [np.mean(f**2) for f in frames]
    sorted_powers = sorted(frame_powers)
    noise_power = np.mean(sorted_powers[: max(1, len(sorted_powers) // 10)])

    signal_power = np.mean(audio_data**2)

    if noise_power == 0:
        snr = float("inf")
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    print(f"\nAudio Clarity (SNR): {snr:.2f} dB")
    logger.info("Audio SNR: %.2f dB", snr)

    # Generate spectrogram
    _save_spectrogram(audio_data, sample_rate)

    return snr


def _save_spectrogram(audio_data: np.ndarray, sample_rate: int) -> Path:
    """Generate and save a spectrogram visualization."""
    f, t, Sxx = spectrogram(audio_data, fs=sample_rate)

    fig, ax = plt.subplots()
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Avoid log(0)
    ax.pcolormesh(t, f, Sxx_db, shading="gouraud")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title("Audio Spectrogram")

    path = OUTPUT_DIR / "spectrogram.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spectrogram saved to %s", path)
    return path
