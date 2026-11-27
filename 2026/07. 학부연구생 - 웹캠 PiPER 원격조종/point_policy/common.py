"""point_policy 공용 (2026-11-27). 교수님이 말씀하신 논문 Point Policy (Haldar·Pinto, arXiv 2502.20391) 를 PiPER 시뮬에 맞춘 것.

webcam_piper (MediaPipe 실시간 조종) 는 그대로 두고, 카메라 보정(A4 종이)·삼각측량·PiPER 운동학만 읽어서 쓴다 (고치지 않음).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEBCAM = ROOT / "webcam_piper"
if str(WEBCAM) not in sys.path:
    sys.path.append(str(WEBCAM))
DATA = HERE / "data"
TABLE_Z = 0.90


def load_task(path=None):
    p = Path(path) if path else HERE / "task_cube.json"
    if not p.is_absolute() and not p.exists():
        p = HERE / p
    t = json.loads(p.read_text(encoding="utf-8"))
    t["_path"] = str(p)
    return t
