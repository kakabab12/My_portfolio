"""data/<label>/*.npy 로 "손등팔등" 분류기를 학습시켜 ONNX로 export한다.

collect_landmarks.py가 녹화한 원본 픽셀 랜드마크를 hand_pose_classifier의
normalize_landmarks()로 정규화해 42차원 벡터로 만들고, 소형 PyTorch MLP로
손등팔등/none 2클래스를 분류하도록 학습한다.

사용법:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 150 --batch_size 32

결과물: models/weights/hand_pose_classifier.onnx (softmax 포함, 추론 시 그대로 확률),
        models/weights/hand_pose_label_map.json (라벨 목록 등 메타데이터)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.inference.hand_pose_classifier import FEATURE_DIM, LABELS, normalize_landmarks  # noqa: E402

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "weights")
ONNX_PATH = os.path.join(MODELS_DIR, "hand_pose_classifier.onnx")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "hand_pose_label_map.json")
HIDDEN_DIM = 32


def load_dataset():
    """data/<label>/*.npy(각 (n_frames, 21, 2) 원본 픽셀좌표)를 정규화된 (X, y)로 합친다."""
    features, labels = [], []
    counts = {}
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            counts[label] = 0
            continue
        n_frames = 0
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".npy"):
                continue
            clip = np.load(os.path.join(label_dir, fname))  # (n_frames, 21, 2)
            for frame_landmarks in clip:
                features.append(normalize_landmarks(frame_landmarks))
                labels.append(label_idx)
            n_frames += len(clip)
        counts[label] = n_frames

    print("클래스별 샘플(프레임) 수:")
    for label, count in counts.items():
        flag = "  <- 너무 적음, 더 녹화 권장" if 0 < count < 30 else ("  <- 없음!" if count == 0 else "")
        print(f"  {label:6s}: {count}{flag}")

    if not features:
        raise SystemExit(
            "학습 데이터가 없습니다. 먼저 python scripts/collect_landmarks.py 로 녹화하세요."
        )
    return np.stack(features), np.array(labels, dtype=np.int64)


class HandPoseMLP:
    """지연 임포트용 팩토리 — 모듈 최상단에서 torch를 강제로 물지 않기 위함."""

    @staticmethod
    def build(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=len(LABELS)):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )


def _confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    X, y = load_dataset()

    n_val = max(1, int(len(X) * args.val_ratio))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_X, val_y = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = HandPoseMLP.build()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X).argmax(dim=1)
                val_acc = (val_pred == val_y).float().mean().item()
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={total_loss / len(train_ds):.4f}  val_acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(val_X).argmax(dim=1).numpy()
    matrix = _confusion_matrix(val_y.numpy(), val_pred, len(LABELS))
    print("\nconfusion matrix (행=정답, 열=예측):")
    print("        " + " ".join(f"{l[:5]:>5s}" for l in LABELS))
    for i, label in enumerate(LABELS):
        print(f"{label:6s}  " + " ".join(f"{v:5d}" for v in matrix[i]))
        support = matrix[i].sum()
        recall = matrix[i, i] / support if support else 0.0
        if support and recall < 0.7:
            print(f"        (recall {recall:.2f} — {label} 데이터를 더 녹화하고 재학습 권장)")

    export_onnx(model)
    save_label_map()
    print(f"\n완료: {ONNX_PATH}\n      {LABEL_MAP_PATH}")


def export_onnx(model):
    import torch
    import torch.nn as nn

    os.makedirs(MODELS_DIR, exist_ok=True)
    export_model = nn.Sequential(model, nn.Softmax(dim=-1))  # 추론 시 바로 확률이 나오게
    export_model.eval()
    dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy,
        ONNX_PATH,
        input_names=["landmarks"],
        output_names=["probs"],
        dynamic_axes={"landmarks": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )


def save_label_map():
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "feature_dim": FEATURE_DIM, "hidden_dim": HIDDEN_DIM}, f,
                  ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
