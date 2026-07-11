"""수집된 랜드마크 시퀀스(.npy)를 읽어 PyTorch Dataset으로 제공."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from augment import augment_sequence
from config import GESTURES, SEQ_LEN
from landmarks import normalize_frame, resample_sequence


def load_dataset(data_dir: Path):
    """data_dir/<gesture>/<person>_<gesture>_<rep>_<ts>.npy 를 전부 읽어 정규화 후
    (samples, persons) 를 반환. samples[i] = (normalized_seq[T_i, FRAME_DIM], label_idx)."""
    label_to_idx = {g: i for i, g in enumerate(GESTURES)}
    samples = []
    persons = []
    for gesture in GESTURES:
        gdir = data_dir / gesture
        if not gdir.exists():
            continue
        for f in sorted(gdir.glob("*.npy")):
            raw = np.load(f)  # (T, FRAME_DIM)
            if raw.ndim != 2 or len(raw) < 3:
                continue
            norm = np.stack([normalize_frame(v) for v in raw])
            samples.append((norm, label_to_idx[gesture]))
            persons.append(f.stem.split("_")[0])
    return samples, persons


class GestureSequenceDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        seq = resample_sequence(seq, SEQ_LEN, jitter=self.train)
        if self.train:
            seq = augment_sequence(seq)
        return torch.from_numpy(seq), label
