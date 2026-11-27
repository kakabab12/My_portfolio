"""책상 높이 재기 (11/23). CAN 을 듣기만 하므로 팔은 움직이지 않는다.

    python table_probe.py              # 6분 동안 듣기 (Ctrl+C 로 일찍 끝내도 그때까지 잰 것을 저장)

11/23 B 가 책상에 박은 뒤, 힘이 빠져 책상에 얹힌 집게 끝이 가상 팔 모델로 책상 +0.9 cm 였다. 모델의 책상 높이(piper_kin.TABLE_Z)가
실제와 1 cm 쯤 다르다는 뜻이라 직접 잰다.

하는 법: robot_server 를 끄고, 팔을 교시 모드(손으로 끌기)로 둔다. 그리퍼를 닫고 집게가 똑바로 아래를 보게 한 채,
집게 끝을 책상에 살짝 대고 2초 가만히 둔다. 자리를 옮겨 (3 cm 넘게) 또 댄다. 밑동 가까이, 멀리, 왼쪽, 오른쪽, 가운데 5~8곳.
2초 동안 관절이 0.15° 안으로 가만히 있고 집게 끝이 모델로 책상 위 5 cm 안이면 한 곳으로 적는다 (공중에서 멈춘 것은 거른다).
끝나면 모델 책상 높이와의 차를 평면(높이 = a + b·앞 + c·옆)으로 맞춰 보여 주고 data/table_probe_시각.json 에 남긴다.
"""
import argparse
import datetime as dt
import json
import math
import os
import time
from collections import deque

import numpy as np

from piper_kin import TABLE_Z, PiperKin
from save_home import CTRL, listen

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_plane(pts):
    """pts: N×3 (앞 x, 옆 y, 모델 높이 − 책상). 돌려줌 (a, b, c, 남은 오차 표준편차)."""
    P = np.asarray(pts, np.float64)
    if len(P) < 3:
        return float(P[:, 2].mean()), 0.0, 0.0, float(P[:, 2].std())
    A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    res = P[:, 2] - A @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(res.std())


def main():
    ap = argparse.ArgumentParser(description="집게 끝을 책상에 대 가며 모델 책상 높이와의 차를 잰다 (CAN 듣기만)")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--still", type=float, default=2.0, help="이만큼(초) 가만히 있으면 한 곳으로 적음")
    ap.add_argument("--max-tilt", type=float, default=15.0, help="집게가 아래에서 이만큼(도) 넘게 기울면 안 적음")
    a = ap.parse_args()
    kin = PiperKin()
    hist = deque()
    spots, warned = [], 0.0
    t_end = time.time() + a.minutes * 60
    print(f"[책상 재기] {a.minutes:.0f}분 동안 듣습니다. 집게 끝을 책상에 대고 {a.still:.0f}초 가만히 두세요. 끝은 Ctrl+C", flush=True)
    try:
        while time.time() < t_end:
            samples, grip, status = listen(a.can, 0.25)
            if not samples:
                print("[책상 재기] 관절 값이 안 옴 (팔 전원, CAN)", flush=True)
                time.sleep(1.0)
                continue
            now = time.time()
            for q in samples[-3:]:
                hist.append((now, np.asarray(q, np.float64)))
            while hist and now - hist[0][0] > a.still + 0.3:
                hist.popleft()
            if len(hist) < 10 or now - hist[0][0] < a.still:
                continue
            Q = np.array([h[1] for h in hist])
            if float((Q.max(0) - Q.min(0)).max()) > 0.15:
                continue
            q = np.radians(Q.mean(0))
            p, m = kin.fk(q)
            h = float(p[2] - TABLE_Z)
            tilt = math.degrees(math.acos(float(np.clip(-m[2, 2], -1.0, 1.0))))
            if h > 0.05:
                continue
            if any(np.linalg.norm(p[:2] - np.array(s["xy"])) < 0.03 for s in spots):
                continue
            if tilt > a.max_tilt:
                if now - warned > 5.0:
                    print(f"[책상 재기] 집게가 {tilt:.0f}° 기울어 있어 안 적음. 똑바로 아래로", flush=True)
                    warned = now
                continue
            mode = CTRL.get(status[0], hex(status[0])) if status else "?"
            spots.append({"xy": [round(float(p[0]), 4), round(float(p[1]), 4)], "h_model": round(h, 4),
                          "tilt_deg": round(tilt, 1), "grip_m": None if grip is None else round(grip, 5),
                          "q_deg": [round(float(v), 3) for v in Q.mean(0)], "mode": mode})
            print(f"[{len(spots)}] 밑동 앞 {p[0] * 100:5.1f} cm, 왼쪽 {p[1] * 100:+5.1f} cm → 모델로 책상 위 {h * 100:+.1f} cm "
                  f"(집게 기울기 {tilt:.0f}°, 그리퍼 {(grip or 0) * 1000:.0f} mm, 모드 {mode})", flush=True)
    except KeyboardInterrupt:
        pass
    if not spots:
        print("[책상 재기] 적은 곳이 없음", flush=True)
        return
    P = np.array([s["xy"] + [s["h_model"]] for s in spots])
    a0, b, c, res = fit_plane(P)
    print(f"\n[결과] {len(spots)}곳. 집게 끝이 책상에 닿았을 때 모델 높이: 평균 {P[:, 2].mean() * 100:+.1f} cm, "
          f"가장 낮은 {P[:, 2].min() * 100:+.1f}, 가장 높은 {P[:, 2].max() * 100:+.1f} cm", flush=True)
    if len(spots) >= 3:
        print(f"[결과] 평면으로 맞추면 밑동 자리 {a0 * 100:+.1f} cm, 앞으로 10 cm 마다 {b * 10:+.2f} cm, 왼쪽으로 10 cm 마다 {c * 10:+.2f} cm, "
              f"남은 오차 {res * 1000:.1f} mm", flush=True)
    out = os.path.join(HERE, "data", "table_probe_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"table_z_model": TABLE_Z, "spots": spots,
                   "plane": {"a_m": a0, "b_per_m": b, "c_per_m": c, "res_m": res}}, f, ensure_ascii=False, indent=1)
    print(f"[저장] {out}", flush=True)


if __name__ == "__main__":
    main()
