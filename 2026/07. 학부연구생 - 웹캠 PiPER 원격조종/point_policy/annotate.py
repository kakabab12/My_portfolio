"""물체 점 찍기 — 시범 하나의 첫 장면에서 (논문: 사람이 몇 초 만에 한 번만, 2026-11-27).

작업 설정(task_cube.json)의 object_points 순서대로, 카메라 0 화면에서 한 번, 카메라 1 화면에서 한 번 누른다.
(정육면체: 윗면 가운데, 판: 가운데). 나머지 시범의 첫 장면에는 object_points.py 가 DINOv2 로 같은 점을 찾아 옮긴다.
옮긴 점이 틀리면 --ep 로 그 시범만 직접 찍을 수 있다 (그 시범에는 찍은 점을 그대로 씀).
키: 왼쪽 클릭 = 점, U = 하나 되돌리기, Enter = 저장, Q = 그만.

  ~/smolVLA/webcam_piper/.venv/bin/python point_policy/annotate.py point_policy/data/cube_to_pad/real [--ep ep_003]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import common  # noqa: F401
from common import load_task
from episode_io import list_episodes

DISP_W = 960


def first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{path} 를 못 읽음")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ep", default=None, help="기본: 첫 시범")
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    task = load_task(a.task)
    folder = Path(a.folder)
    eps = list_episodes(folder)
    ep = folder / a.ep if a.ep else eps[0]
    names = [op["name"] for op in task["object_points"]]
    labels = [op["label"] for op in task["object_points"]]
    imgs = [first_frame(ep / f"cam{c}.avi") for c in range(2)]
    clicks = {0: [], 1: []}
    order = [(c, i) for i in range(len(names)) for c in range(2)]
    s = DISP_W / imgs[0].shape[1]
    win = "annotate object points"
    cv2.namedWindow(win)
    state = {"cam": 0}

    def on_mouse(ev, x, y, *_):
        n = len(clicks[0]) + len(clicks[1])
        if ev == cv2.EVENT_LBUTTONDOWN and n < len(order):
            c, _i = order[n]
            clicks[c].append([x / s, y / s])
    cv2.setMouseCallback(win, on_mouse)
    while True:
        n = len(clicks[0]) + len(clicks[1])
        c = order[n][0] if n < len(order) else 1
        state["cam"] = c
        img = cv2.resize(imgs[c], (DISP_W, int(imgs[c].shape[0] * s)))
        for k, p in enumerate(clicks[c]):
            cv2.drawMarker(img, (int(p[0] * s), int(p[1] * s)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(img, names[k], (int(p[0] * s) + 8, int(p[1] * s) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if n < len(order):
            t = f"cam{c}: click  [{labels[order[n][1]]}]   (U undo)"
        else:
            t = "done. Enter = save, U = undo"
        cv2.putText(img, f"{ep.name}  " + t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("u") and n:
            cc, _ = order[n - 1]
            clicks[cc].pop()
        if k in (13, 10) and n == len(order):
            break
    cv2.destroyAllWindows()
    path = folder / "annotations.json"
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"names": names, "episodes": {}}
    d["names"] = names
    d.setdefault("reference", ep.name)
    if not a.ep:
        d["reference"] = ep.name
    d["episodes"][ep.name] = {"cam0": clicks[0], "cam1": clicks[1]}
    path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"→ {path} ({ep.name}, 기준 시범 {d['reference']})")


if __name__ == "__main__":
    main()
