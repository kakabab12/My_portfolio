"""Continuously classify microphone audio with a trained SVM model."""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import joblib
import numpy as np
import sounddevice as sd

from audio_utils import anomaly_probability, fixed_clip, mfcc_feature, resample_audio


def native_sample_rate(device: int | str | None) -> int:
    return int(round(sd.query_devices(device, "input")["default_samplerate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live normal/abnormal prediction from a microphone.")
    parser.add_argument("--model", type=Path, default=Path("models/fan_svm.joblib"))
    parser.add_argument("--device", help="Input device number or exact name; omit for default")
    parser.add_argument("--capture-sample-rate", type=int, help="Default: microphone native sample rate")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between displayed predictions")
    parser.add_argument("--threshold", type=float, help="Default: threshold saved in the model")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}. Run train_fan.py first.")
    if args.channels < 1 or args.interval <= 0:
        raise SystemExit("Check --channels and --interval values.")

    model = joblib.load(args.model)
    threshold = args.threshold if args.threshold is not None else float(model.get("threshold", 0.50))
    if not 0 < threshold < 1:
        raise SystemExit("--threshold must be between 0 and 1.")
    target_rate = int(model["sample_rate"])
    clip_seconds = float(model["clip_seconds"])
    device: int | str | None = int(args.device) if args.device and args.device.isdigit() else args.device
    capture_rate = args.capture_sample_rate or native_sample_rate(device)
    window_samples = int(round(capture_rate * clip_seconds))
    hop_samples = int(round(capture_rate * args.interval))
    chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=40)

    def callback(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            print(f"[audio status] {status}")
        mono = np.mean(indata, axis=1, dtype=np.float32)
        try:
            chunks.put_nowait(mono.copy())
        except queue.Full:
            pass

    print(f"Live {model.get('machine', 'machine')} prediction started. Press Ctrl+C to stop.")
    print(f"Capture: {capture_rate} Hz | Model: {target_rate} Hz | Window: {clip_seconds:.1f}s")
    buffer = np.empty(0, dtype=np.float32)
    samples_after_prediction = 0
    try:
        with sd.InputStream(
            samplerate=capture_rate,
            channels=args.channels,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            while True:
                try:
                    incoming = chunks.get(timeout=1.0)
                except queue.Empty:
                    continue
                buffer = np.concatenate((buffer, incoming))
                samples_after_prediction += incoming.size
                if buffer.size > window_samples:
                    buffer = buffer[-window_samples:]
                if buffer.size < window_samples or samples_after_prediction < hop_samples:
                    continue

                samples_after_prediction = 0
                model_audio = resample_audio(buffer, capture_rate, target_rate)
                model_samples = int(round(target_rate * clip_seconds))
                feature = mfcc_feature(fixed_clip(model_audio, model_samples), target_rate, model["n_mfcc"])
                probability = anomaly_probability(model, feature)
                result = "ABNORMAL" if probability >= threshold else "NORMAL"
                print(f"{time.strftime('%H:%M:%S')} | anomaly={probability:.3f} | {result}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
