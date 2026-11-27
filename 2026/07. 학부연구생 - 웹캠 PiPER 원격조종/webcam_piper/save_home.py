"""지금 팔 자세를 처음 자세로 저장한다 (11/20). CAN 을 듣기만 하므로 팔은 움직이지 않는다.

    python save_home.py            # can0 에서 0.5초 듣고 home_pose.json 에 저장
    python save_home.py --show     # 저장하지 않고 지금 자세만 보기

팔을 교시 모드(손으로 끌어 옮기는 모드)로 원하는 자세에 둔 채 실행하면 된다. robot_server 가 돌고 있어도 된다 (듣기만 함).
팔이 움직이는 중이면 (0.5초 동안 0.5° 넘게 바뀜) 저장하지 않는다.
돌아가기는 teleop 화면에서 B (robot_server 가 돌 때). 교시 모드에서는 팔이 명령을 안 받으니, 돌아가기 전에 교시 모드를 끈다.
"""
import argparse
import datetime as dt
import json
import math
import socket
import struct
import time

from home import HOME_FILE

CTRL = {0: "대기", 1: "CAN 명령", 2: "교시(손으로 끌기)", 3: "이더넷", 4: "와이파이", 5: "리모컨", 6: "연동 교시", 7: "오프라인 궤적"}


def listen(can="can0", seconds=0.5):
    """PiPER 가 내보내는 관절 피드백(0x2A5~0x2A7, 0.001° 단위)·그리퍼(0x2A8, 0.001 mm)·상태(0x2A1)를 듣는다."""
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((can,))
    s.settimeout(0.5)
    last, samples, grip, status = {}, [], None, None
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            try:
                frame = s.recv(16)
            except socket.timeout:
                break
            cid, dlc, data = struct.unpack("=IB3x8s", frame)
            cid &= 0x7FF
            last[cid] = data[:dlc]
            if cid == 0x2A8:
                grip = struct.unpack(">i", data[:4])[0] / 1e6            # m
            elif cid == 0x2A1:
                status = data[:dlc]
            elif cid == 0x2A7 and 0x2A5 in last and 0x2A6 in last:
                q = []
                for c in (0x2A5, 0x2A6, 0x2A7):
                    a, b = struct.unpack(">ii", last[c][:8])
                    q += [a / 1000.0, b / 1000.0]
                samples.append(q)
    finally:
        s.close()
    return samples, grip, status


def main():
    ap = argparse.ArgumentParser(description="지금 팔 자세를 처음 자세로 저장 (CAN 듣기만)")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--file", default=HOME_FILE)
    ap.add_argument("--show", action="store_true", help="저장하지 않고 보기만")
    a = ap.parse_args()
    samples, grip, status = listen(a.can)
    if not samples:
        raise SystemExit(f"{a.can} 에서 팔 관절 값을 못 받음 (팔 전원, CAN 켜기: sudo ip link set {a.can} up type can bitrate 1000000)")
    deg = [sum(c) / len(c) for c in zip(*samples)]
    spread = max(max(c) - min(c) for c in zip(*samples))
    mode = CTRL.get(status[0], hex(status[0])) if status else "?"
    print(f"[팔] 관절(도): {', '.join(f'{v:.2f}' for v in deg)}   그리퍼 {grip * 1000:.1f} mm   모드: {mode}"
          f"   ({len(samples)}번 읽음, 흔들림 {spread:.2f}°)")
    try:
        import numpy as np
        from piper_kin import TABLE_Z, PiperKin
        p, _ = PiperKin().fk(np.radians(deg))
        print(f"[팔] 집게 끝: 밑동에서 앞 {p[0] * 100:.1f} cm, 왼쪽 {p[1] * 100:.1f} cm, 책상 위 {(p[2] - TABLE_Z) * 100:.1f} cm (가상 팔 모델 기준)")
        from home import HOME_CLEAR
        if p[2] - TABLE_Z < HOME_CLEAR:
            print(f"[주의] 집게 끝이 책상에 가깝다. B 는 이 자세 그대로가 아니라 같은 방향으로 모델 책상 위 {HOME_CLEAR * 100:.0f} cm 까지 들어 올린 자세로 간다"
                  " (11/23 책상에 박은 일). 책상에서 2~3 cm 띄운 채 저장하면 그대로 간다")
    except Exception as e:  # noqa: BLE001 — numpy 가 없는 파이썬에서도 저장은 되게
        print(f"[팔] 집게 끝 계산 못 함 ({e})")
    if a.show:
        return
    if spread > 0.5:
        raise SystemExit("팔이 움직이는 중이라 저장하지 않음. 팔을 가만히 둔 채 다시 실행")
    out = {"q_rad": [round(math.radians(v), 6) for v in deg], "q_deg": [round(v, 3) for v in deg],
           "grip_m": None if grip is None else round(grip, 5), "mode_when_saved": mode,
           "saved": dt.datetime.now().isoformat(timespec="seconds"),
           "note": "save_home.py 가 CAN 피드백을 듣고 저장. teleop B 로 이 관절 자세로 돌아감 (그리퍼는 11/23 부터 닫힘)"}
    with open(a.file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[저장] {a.file}")


if __name__ == "__main__":
    main()
