"""시범 한 개의 파일 꼴 (진짜 손 녹화와 가짜 시범이 같은 꼴, 2026-11-27).

data/<작업>/<묶음>/ep_000/
  meta.json      fps, 장 수, 카메라 보정 두 개 (webcam_piper TableCalib.to_dict), 어디서 왔나(source)
  cam0.avi cam1.avi  녹화 영상 (진짜 손만, MJPG)
  times.npy      장마다 찍힌 때 (초, 두 카메라 가운데)
  hands2d.npz    cam0, cam1: (T, 21, 2) MediaPipe 손 21점 화면 위치 px (못 찾으면 NaN)
  objects2d.npz  cam0, cam1: (T, K, 2) 물체 점 화면 위치 px, vis0, vis1: (T, K) 보이는지 (CoTracker)
"""
import json
from pathlib import Path

import numpy as np

import common  # noqa: F401
from geometry import TableCalib, proj_matrix


def list_episodes(folder):
    return sorted(p for p in Path(folder).glob("ep_*") if (p / "meta.json").exists())


def new_episode_dir(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    while (folder / f"ep_{n:03d}").exists():
        n += 1
    d = folder / f"ep_{n:03d}"
    d.mkdir()
    return d


def load_meta(ep):
    return json.loads((Path(ep) / "meta.json").read_text(encoding="utf-8"))


def save_meta(ep, meta):
    (Path(ep) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def calibs(meta):
    return [TableCalib.from_dict(meta["calib"][0]), TableCalib.from_dict(meta["calib"][1])]


def proj_matrices(meta):
    return [proj_matrix(c) for c in calibs(meta)]
