"""Shared audio loading, MFCC extraction, and model prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio_files(folder: Path) -> list[Path]:
    """Return supported audio files directly under *folder*, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_mono(path: Path | str, sample_rate: int) -> np.ndarray:
    """Load audio as a finite, mono float32 signal at the requested sample rate."""
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if audio.size == 0:
        raise ValueError(f"Audio has no samples: {path}")
    return audio


def clip_offsets(num_samples: int, clip_samples: int, max_clips: int) -> list[int]:
    """Choose evenly spaced clip starts. Short files still produce one padded clip."""
    if num_samples <= clip_samples or max_clips <= 1:
        return [0]
    last_start = num_samples - clip_samples
    return sorted({int(value) for value in np.linspace(0, last_start, max_clips)})


def fixed_clip(audio: np.ndarray, clip_samples: int, start: int = 0) -> np.ndarray:
    """Extract one fixed-size clip and zero-pad the end if necessary."""
    segment = audio[start : start + clip_samples]
    if segment.size < clip_samples:
        segment = np.pad(segment, (0, clip_samples - segment.size))
    return np.asarray(segment, dtype=np.float32)


def mfcc_feature(audio: np.ndarray, sample_rate: int, n_mfcc: int = 20) -> np.ndarray:
    """Create a compact MFCC feature vector: MFCC, delta, and delta-delta statistics."""
    if audio.size < 512:
        audio = np.pad(audio, (0, 512 - audio.size))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    statistics: list[np.ndarray] = []
    for matrix in (mfcc, delta, delta2):
        statistics.extend((np.mean(matrix, axis=1), np.std(matrix, axis=1)))
    return np.concatenate(statistics).astype(np.float32)


def extract_file_features(
    path: Path | str,
    sample_rate: int,
    clip_seconds: float,
    max_clips: int,
    n_mfcc: int,
) -> np.ndarray:
    """Load a file and return one MFCC feature vector per evenly spaced clip."""
    audio = load_mono(path, sample_rate)
    clip_samples = max(512, int(round(sample_rate * clip_seconds)))
    offsets = clip_offsets(audio.size, clip_samples, max_clips)
    return np.vstack(
        [mfcc_feature(fixed_clip(audio, clip_samples, offset), sample_rate, n_mfcc) for offset in offsets]
    )


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=original_rate, target_sr=target_rate)


def anomaly_probability(model: dict, feature: np.ndarray) -> float:
    """Return P(abnormal) for one feature vector from a saved fan model."""
    probabilities = model["pipeline"].predict_proba(np.asarray(feature, dtype=np.float32).reshape(1, -1))[0]
    classes = list(model["pipeline"].classes_)
    return float(probabilities[classes.index("abnormal")])
