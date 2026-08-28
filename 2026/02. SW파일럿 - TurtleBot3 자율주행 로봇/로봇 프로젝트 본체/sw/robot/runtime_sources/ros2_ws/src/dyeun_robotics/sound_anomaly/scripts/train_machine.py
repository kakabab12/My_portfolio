"""Train one MIMII machine model and verify it on a separate target domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_utils import extract_file_features


LABELS = ("normal", "abnormal")


def label_for(path: Path) -> str:
    if "_normal_" in path.name:
        return "normal"
    if "_anomaly_" in path.name:
        return "abnormal"
    raise ValueError(f"Cannot determine label from {path.name}")


def labelled_files(folder: Path) -> list[tuple[Path, str]]:
    files = [(path, label_for(path)) for path in sorted(folder.glob("*.wav"))]
    counts = {label: sum(file_label == label for _, file_label in files) for label in LABELS}
    if not all(counts.values()):
        raise ValueError(f"Both classes are required in {folder}: {counts}")
    return files


def training_matrix(
    files: list[tuple[Path, str]], sample_rate: int, clip_seconds: float, clips_per_file: int, n_mfcc: int
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for index, (path, label) in enumerate(files, start=1):
        features = extract_file_features(path, sample_rate, clip_seconds, clips_per_file, n_mfcc)
        vectors.append(features)
        labels.extend([label] * len(features))
        if index % 100 == 0 or index == len(files):
            print(f"  extracted {index}/{len(files)} training files")
    return np.vstack(vectors), np.asarray(labels)


def evaluate_files(
    pipeline: Pipeline,
    files: list[tuple[Path, str]],
    sample_rate: int,
    clip_seconds: float,
    clips_per_file: int,
    n_mfcc: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth: list[str] = []
    predicted: list[str] = []
    probabilities: list[float] = []
    classes = list(pipeline.classes_)
    abnormal_index = classes.index("abnormal")
    for index, (path, label) in enumerate(files, start=1):
        features = extract_file_features(path, sample_rate, clip_seconds, clips_per_file, n_mfcc)
        probability = float(np.mean(pipeline.predict_proba(features)[:, abnormal_index]))
        truth.append(label)
        predicted.append("abnormal" if probability >= threshold else "normal")
        probabilities.append(probability)
        if index % 100 == 0 or index == len(files):
            print(f"  evaluated {index}/{len(files)} target files")
    return np.asarray(truth), np.asarray(predicted), np.asarray(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-machine normal/abnormal model for Jetson inference.")
    parser.add_argument("--machine", choices=("fan", "gearbox", "pump"), default="gearbox")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--clips-per-file", type=int, default=3)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    machine_root = args.root / args.machine
    model_out = args.model_out or args.root / "models" / f"{args.machine}_svm_source.joblib"
    report_out = args.report_out or args.root / "models" / f"{args.machine}_source_to_target_report.json"
    source_files = labelled_files(machine_root / "source_test")
    target_files = labelled_files(machine_root / "target_test")
    print(f"Training {args.machine} model on {len(source_files)} source files...")
    x_train, y_train = training_matrix(
        source_files, args.sample_rate, args.clip_seconds, args.clips_per_file, args.n_mfcc
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    print(f"Evaluating on {len(target_files)} completely separate target files...")
    y_true, y_pred, probabilities = evaluate_files(
        pipeline,
        target_files,
        args.sample_rate,
        args.clip_seconds,
        args.clips_per_file,
        args.n_mfcc,
        args.threshold,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["abnormal"], average=None, zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    report = {
        "machine": args.machine,
        "method": "Train on source_test; evaluate only on target_test; average three 3-second predictions per WAV.",
        "source_training_files": len(source_files),
        "target_evaluation_files": len(target_files),
        "file_accuracy": round(accuracy, 4),
        "abnormal_precision": round(float(precision[0]), 4),
        "abnormal_recall": round(float(recall[0]), 4),
        "abnormal_f1": round(float(f1[0]), 4),
        "threshold": args.threshold,
        "confusion_matrix_normal_abnormal": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0),
        "probability_range": [round(float(probabilities.min()), 4), round(float(probabilities.max()), 4)],
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "sample_rate": args.sample_rate,
            "clip_seconds": args.clip_seconds,
            "clips_per_file": args.clips_per_file,
            "n_mfcc": args.n_mfcc,
            "labels": LABELS,
            "threshold": args.threshold,
            "machine": args.machine,
            "description": f"{args.machine} normal/abnormal MFCC + RBF SVM, trained on MIMII source_test",
        },
        model_out,
    )
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFile accuracy: {accuracy:.1%}")
    print(f"Abnormal precision/recall/F1: {precision[0]:.1%} / {recall[0]:.1%} / {f1[0]:.1%}")
    print(f"Saved model: {model_out.resolve()}")
    print(f"Saved report: {report_out.resolve()}")


if __name__ == "__main__":
    main()
