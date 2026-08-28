"""Classify one audio file using a trained normal/abnormal model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from audio_utils import anomaly_probability, extract_file_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict normal/abnormal for an audio file.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/fan_svm.joblib"))
    parser.add_argument("--clips", type=int, help="Number of evenly spaced clips to inspect (default: model setting)")
    parser.add_argument("--threshold", type=float, help="Abnormal probability threshold (default: model setting)")
    args = parser.parse_args()

    if not args.audio_path.is_file():
        raise SystemExit(f"Audio file not found: {args.audio_path}")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}. Run train_fan.py first.")
    model = joblib.load(args.model)
    clips = args.clips if args.clips is not None else int(model.get("clips_per_file", 5))
    threshold = args.threshold if args.threshold is not None else float(model.get("threshold", 0.50))
    if clips < 1 or not 0 < threshold < 1:
        raise SystemExit("--clips must be positive and --threshold must be between 0 and 1.")
    features = extract_file_features(
        args.audio_path,
        model["sample_rate"],
        model["clip_seconds"],
        clips,
        model["n_mfcc"],
    )
    probabilities = np.array([anomaly_probability(model, feature) for feature in features])
    probability = float(np.mean(probabilities))
    result = "ABNORMAL" if probability >= threshold else "NORMAL"

    print(f"File             : {args.audio_path}")
    print(f"Clips analysed   : {len(probabilities)}")
    print(f"Anomaly scores   : {', '.join(f'{item:.3f}' for item in probabilities)}")
    print(f"Mean anomaly prob: {probability:.3f}")
    print(f"Threshold        : {threshold:.2f}")
    print(f"Result           : {result}")


if __name__ == "__main__":
    main()
