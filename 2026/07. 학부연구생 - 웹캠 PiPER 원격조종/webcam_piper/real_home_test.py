"""실물 B 시험 (11/20): 카메라 없이 가짜 손 값을 robot_server 로 보내 집게 끝을 옮긴 뒤, B 로 처음 자세에 돌아오는지 본다.

    (robot_server.py --robot agxbus ... 를 먼저 켜 둔 채)  .venv/bin/python real_home_test.py

1. 따라가기 끄고 1초 (서버가 '처음엔 꺼져 있었음' 을 알게)
2. 따라가기 켜고 4초 동안 가짜 손을 지금 집게 끝 자리 → 밑동 앞 25 cm, 책상 위 10 cm 로 천천히 (최소 저크). 그리퍼 폭은 지금 그대로, 집게 방향은 안 바꿈
3. 그 자리에서 2초
4. 따라가기 끄고 B → 처음 자세로 (서버가 1.5~2초에 걸쳐 감), 5초 기다림
서버 답장(팔 관절)을 0.05초마다 남겨 data/real_home_test_날짜.json 에. 중간에 멈추려면 이 창 Ctrl+C (서버가 0.5초 뒤 그 자리에 멈춤).
"""
import argparse
import datetime as dt
import json
import os
import time

import numpy as np

from home import load_home
from net import PORT, Link
from piper_kin import TABLE_Z, PiperKin

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL = np.array([0.25, 0.0, 0.10])          # 로봇 좌표: 밑동 앞 25 cm, 가운데, 책상 위 10 cm
X0 = 0.30                                    # robot_server 기본 --x0
SQUEEZE = 0.025


def mj(s):
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT, help="가상 팔로 미리 돌려 볼 때 다른 포트")
    a = ap.parse_args()
    kin, home = PiperKin(), load_home()
    link = Link(f"127.0.0.1:{a.port}")
    log, T0 = [], time.monotonic()
    state = {"q": None, "w": None, "home_n": 0}

    def tick(follow, hand, gap):
        link.send(hand=None if hand is None else [round(float(v), 5) for v in hand] + [0.0, round(gap, 4)],
                  follow=int(follow), rec=0, grip=1, yawf=1, sq=SQUEEZE, ygm=0.05, ypref=-1.047,
                  disc=0, home=state["home_n"], bye=0, gm=0, src=2)
        st = link.poll()
        if st:
            state["q"], state["w"] = st["q"], st.get("w")
            log.append({"t": round(time.monotonic() - T0, 3), "q": st["q"], "w": st.get("w"), "follow": st["follow"],
                        "homing": st.get("homing"), "msg": st.get("msg")})
            if st.get("msg") and (len(log) < 2 or log[-2].get("msg") != st.get("msg")):
                print(f"  [{time.monotonic() - T0:5.1f}s] 서버: {st['msg']}")
        time.sleep(1 / 23)

    try:
        print("1. 따라가기 끈 채 연결 (1초)")
        t = time.monotonic()
        while time.monotonic() - t < 1.0 or state["q"] is None:
            tick(False, None, 0.0)
            if time.monotonic() - t > 5:
                raise SystemExit("robot_server 답이 없음 (켜져 있는지 확인)")
        q0 = np.array(state["q"])
        p0, _ = kin.fk(q0)
        w0 = state["w"] if state["w"] is not None else 0.01
        start = np.array([p0[0] - X0, p0[1], max(p0[2] - TABLE_Z, 0.012) + 0.0])   # 사람 좌표 = 로봇 − 시작점 (+높이 기준 같음)
        goal = np.array([GOAL[0] - X0, GOAL[1], GOAL[2]])
        gap = float(w0) + SQUEEZE                                                  # 그리퍼 폭 그대로 (gap < 5 cm 라 방향도 안 바꿈)
        print(f"   지금 집게 끝: 앞 {p0[0] * 100:.1f}, 왼 {p0[1] * 100:.1f}, 책상 위 {(p0[2] - TABLE_Z) * 100:.1f} cm, 그리퍼 {w0 * 1000:.1f} mm")
        print("2. 따라가기 켜고 4초에 걸쳐 앞 25 cm, 책상 위 10 cm 로")
        t = time.monotonic()
        while time.monotonic() - t < 4.0:
            s = (time.monotonic() - t) / 4.0
            tick(True, start + (goal - start) * mj(s), gap)
        print("3. 그 자리 2초")
        t = time.monotonic()
        while time.monotonic() - t < 2.0:
            tick(True, goal, gap)
        q_far = np.array(state["q"])
        pf, _ = kin.fk(q_far)
        print(f"   도착: 앞 {pf[0] * 100:.1f}, 왼 {pf[1] * 100:.1f}, 책상 위 {(pf[2] - TABLE_Z) * 100:.1f} cm")
        print("4. 따라가기 끄고 B")
        state["home_n"] += 1
        t = time.monotonic()
        while time.monotonic() - t < 5.0:
            tick(False, None, 0.0)
        q_end = np.array(state["q"])
        err = np.degrees(np.abs(q_end - home)) if home is not None else None
        print("   처음 자세와 관절 차(도):", None if err is None else ", ".join(f"{v:.2f}" for v in err))
    finally:
        for _ in range(5):
            link.send(hand=None, follow=0, rec=0, grip=1, yawf=1, sq=SQUEEZE, disc=0, home=state["home_n"], bye=1)
            time.sleep(0.02)
        link.close()
        out = os.path.join(HERE, "data", f"real_home_test_{dt.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(out, "w") as f:
            json.dump({"home": None if home is None else home.tolist(), "log": log}, f)
        print("기록:", out)


if __name__ == "__main__":
    main()
