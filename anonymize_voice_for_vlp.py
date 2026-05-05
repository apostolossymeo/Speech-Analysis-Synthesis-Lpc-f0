#!/usr/bin/env python3
"""
Create anonymized versions of a speech recording for a Voice & Language Processing project.

Goal: keep timing, energy contours, pauses, and broad spectral structure usable for
spectrogram / voiced-unvoiced / F0 / LPC visualizations, while making the speaker's
identity much harder to recognize.

Usage:
    python anonymize_voice_for_vlp.py total_human.wav

Outputs:
    total_human_anonymized.wav          # recommended replacement for GitHub/project visuals
    total_human_anonymized_robotic.wav  # stronger anonymization, less natural
    total_human_anonymized_low.wav      # lower-pitched alternative
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, lfilter


def normalize(y: np.ndarray, peak: float = 0.96) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > 0:
        y = y / m * peak
    return y.astype(np.float32)


def match_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y


def spectral_tilt(y: np.ndarray, sr: int) -> np.ndarray:
    """Mild filtering to move timbre away from the original voice."""
    # high-pass remove rumble
    b, a = butter(2, 80 / (sr / 2), btype="highpass")
    y = filtfilt(b, a, y).astype(np.float32)
    # pre-emphasis-like tilt
    y = lfilter([1.0, -0.86], [1.0], y).astype(np.float32)
    return y


def add_tiny_noise(y: np.ndarray, amount: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (y + amount * rng.standard_normal(len(y))).astype(np.float32)


def anonymized_recommended(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Recommended balance:
    - pitch shifted up by 5 semitones
    - timbre tilted
    - tiny noise floor
    - same duration as source
    """
    z = librosa.effects.pitch_shift(y, sr=sr, n_steps=5.0)
    z = match_length(z, len(y))
    z = spectral_tilt(z, sr)
    z = add_tiny_noise(z, 0.0015)
    return normalize(z)


def anonymized_low(y: np.ndarray, sr: int) -> np.ndarray:
    """Lower-pitched variant, useful if the high version sounds too cartoonish."""
    z = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4.0)
    z = match_length(z, len(y))
    # Low-pass a little to change brightness.
    b, a = butter(3, min(3600, sr * 0.45) / (sr / 2), btype="lowpass")
    z = filtfilt(b, a, z).astype(np.float32)
    z = add_tiny_noise(z, 0.0015)
    return normalize(z)


def anonymized_robotic(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Stronger anonymization. Ring-modulated / robotic tone.
    Preserves timing and rough energy but sounds much less like the speaker.
    """
    t = np.arange(len(y), dtype=np.float32) / sr
    carrier = 0.62 + 0.38 * np.sin(2 * np.pi * 90 * t)
    z = y * carrier
    z = librosa.effects.pitch_shift(z, sr=sr, n_steps=7.0)
    z = match_length(z, len(y))
    z = spectral_tilt(z, sr)
    return normalize(z)


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("total_human.wav")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    y, sr = librosa.load(in_path, sr=None, mono=True)
    y = normalize(y, peak=0.90)

    out_dir = in_path.parent
    outputs = {
        "total_human_anonymized.wav": anonymized_recommended(y, sr),
        "total_human_anonymized_robotic.wav": anonymized_robotic(y, sr),
        "total_human_anonymized_low.wav": anonymized_low(y, sr),
    }
    for name, audio in outputs.items():
        sf.write(out_dir / name, audio, sr, subtype="PCM_16")
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
