"""python train.py 로 실행. data/ 아래 녹화된 시퀀스를 학습해서 models/gesture_model.pt 를 만듦."""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import DATA_DIR, MODELS_DIR, GESTURES, SEQ_LEN, RANDOM_SEED
from dataset import GestureSequenceDataset, load_dataset
from landmarks import FRAME_DIM
from model import GestureGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    samples, persons = load_dataset(DATA_DIR)
    if not samples:
        raise SystemExit(f"{DATA_DIR} 에 학습 데이터가 없습니다. 먼저 collect_data.py로 녹화하세요.")

    labels = [lbl for _, lbl in samples]
    counts = np.bincount(labels, minlength=len(GESTURES))
    print(f"총 샘플 수: {len(samples)}  (촬영 인원: {sorted(set(persons))})")
    print("클래스별 샘플 수:")
    for g, c in zip(GESTURES, counts):
        print(f"  {g:12s}: {c}")
    missing = [g for g, c in zip(GESTURES, counts) if c == 0]
    if missing:
        print(f"경고: 데이터가 아예 없는 클래스가 있습니다 -> {missing}")

    idx = np.arange(len(samples))
    try:
        train_idx, val_idx = train_test_split(
            idx, test_size=args.val_ratio, random_state=RANDOM_SEED, stratify=labels
        )
    except ValueError:
        print("일부 클래스 샘플이 너무 적어 stratify 없이 분할합니다 (클래스마다 최소 2개 이상 녹화 권장).")
        train_idx, val_idx = train_test_split(idx, test_size=args.val_ratio, random_state=RANDOM_SEED)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_loader = DataLoader(GestureSequenceDataset(train_samples, train=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GestureSequenceDataset(val_samples, train=False), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureGRU(input_dim=FRAME_DIM, num_classes=len(GESTURES)).to(device)

    class_weights = torch.tensor(
        len(labels) / (len(GESTURES) * np.maximum(counts, 1)), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    MODELS_DIR.mkdir(exist_ok=True)
    best_acc = 0.0
    best_preds, best_labels = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / max(len(train_loader.dataset), 1)

        model.eval()
        correct, total = 0, 0
        epoch_preds, epoch_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                epoch_preds += pred.cpu().tolist()
                epoch_labels += y.cpu().tolist()
        val_acc = correct / max(total, 1)
        scheduler.step(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_preds, best_labels = epoch_preds, epoch_labels
            torch.save(model.state_dict(), MODELS_DIR / "gesture_model.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.4f} val_acc={val_acc:.3f} best={best_acc:.3f}")

    label_map = {
        "gestures": GESTURES,
        "seq_len": SEQ_LEN,
        "frame_dim": FRAME_DIM,
        "hidden_dim": 64,
        "num_layers": 2,
    }
    with open(MODELS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n최고 검증 정확도: {best_acc:.3f}")
    print(f"모델 저장 위치: {MODELS_DIR / 'gesture_model.pt'}")

    if best_labels:
        print("\n[최고 성능 시점 기준 confusion matrix] (행=실제, 열=예측)")
        present = sorted(set(best_labels) | set(best_preds))
        print("클래스:", [GESTURES[i] for i in present])
        print(confusion_matrix(best_labels, best_preds, labels=present))
        print(classification_report(
            best_labels, best_preds, labels=present,
            target_names=[GESTURES[i] for i in present], zero_division=0,
        ))


if __name__ == "__main__":
    main()
