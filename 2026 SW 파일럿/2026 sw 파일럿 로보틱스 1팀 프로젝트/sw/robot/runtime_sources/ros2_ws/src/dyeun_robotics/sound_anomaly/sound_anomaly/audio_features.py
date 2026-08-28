"""Audio feature extraction used by the trained gearbox model."""

import librosa
import numpy as np


def fixed_clip(audio: np.ndarray, clip_samples: int) -> np.ndarray:
    segment = audio[-clip_samples:]
    if segment.size < clip_samples:
        segment = np.pad(segment, (clip_samples - segment.size, 0))
    return np.asarray(segment, dtype=np.float32)


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=original_rate,
        target_sr=target_rate,
    )


def mfcc_feature(audio: np.ndarray, sample_rate: int, n_mfcc: int) -> np.ndarray:
    if audio.size < 512:
        audio = np.pad(audio, (0, 512 - audio.size))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
    )
    matrices = (mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2))
    statistics = []
    for matrix in matrices:
        statistics.extend((np.mean(matrix, axis=1), np.std(matrix, axis=1)))
    return np.concatenate(statistics).astype(np.float32)


def anomaly_probability(model: dict, feature: np.ndarray) -> float:
    probabilities = model["pipeline"].predict_proba(feature.reshape(1, -1))[0]
    classes = list(model["pipeline"].classes_)
    return float(probabilities[classes.index("abnormal")])
