"""노트북 점검 (2026-11-16). 사람이 손을 쓰지 않아도 돌려서 확인할 수 있는 것만 본다 (노트북 Claude 가 돌리라고 만듦).

  python check_setup.py                  프로그램·손 모델·계산 시험·웹캠 (웹캠 사진 한 장을 cam_check.jpg 로 남김)
  python check_setup.py --no-cam         웹캠 빼고
  python check_setup.py --robot-pc 주소   연구실 PC 의 robot_server.py 와 주고받기만 한다. 따라가기를 켜지 않으므로 팔은 움직이지 않는다
  python check_setup.py --report         보정과 P 위치 확인 결과를 연구실에 붙여 넣을 한 덩어리로 보여 줌

마지막 줄이 "점검: 모두 됨" 이면 된다. 안 된 것은 "안 됨" 줄에 무엇을 하면 되는지 같이 적는다.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BAD = []


def good(msg):
    print("  됨    " + msg)


def bad(msg, fix):
    print("  안 됨 " + msg + "\n         → " + fix)
    BAD.append(msg)


def versions():
    out = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()} {platform.machine()}"}
    for name in ("numpy", "cv2", "mediapipe"):
        try:
            out[name] = __import__(name).__version__
        except Exception:                                   # 깔리지 않았거나 깨짐
            out[name] = None
    return out


def check_programs():
    v = versions()
    print(f"[1] 프로그램 ({v['os']})")
    if sys.version_info < (3, 9):
        bad(f"Python {v['python']} 은 너무 옛것", "Python 3.10~3.12 로 가상환경을 다시 만든다")
    else:
        good(f"Python {v['python']}")
    missing = [k for k in ("numpy", "cv2", "mediapipe") if v[k] is None]
    if missing:
        bad("없는 것: " + ", ".join(missing), "가상환경을 켠 채로 pip install -r requirements.txt")
        return False
    good(f"numpy {v['numpy']}, opencv {v['cv2']}, mediapipe {v['mediapipe']}")
    import cv2
    info = cv2.getBuildInformation()
    gui = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.strip().startswith("GUI:")), "")
    if gui.upper().startswith("NONE"):
        bad("opencv 가 창을 못 띄우는 판(headless)", "pip uninstall -y opencv-python-headless 뒤 pip install -r requirements.txt")
    else:
        good(f"opencv 창 띄우기 ({gui or '확인 못 함'})")
    return True


def check_model():
    print("[2] 손 모델")
    from hand import HAND_MODEL, HandTracker
    if not os.path.exists(HAND_MODEL):
        bad("models/hand_landmarker.task 가 없음", "python download_models.py")
        return None
    good(f"hand_landmarker.task {os.path.getsize(HAND_MODEL) / 1e6:.1f} MB")
    try:
        import numpy as np
        tr = HandTracker("right")
        res = tr.detect(np.zeros((720, 1280, 3), np.uint8), 1)
        good("MediaPipe 손 찾기가 돎 (빈 화면에서 손 없음 = " + ("맞음" if res is None else "이상함") + ")")
        return tr
    except Exception as e:
        bad(f"MediaPipe 를 못 켬: {type(e).__name__}: {e}", "python download_models.py 를 다시, 그래도 안 되면 Python 3.11·3.12 가상환경")
        return None


def check_math():
    print("[3] 계산 시험 (test_geometry.py, 10초쯤)")
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_geometry.py")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=HERE)
    last = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("결과:")]
    if r.returncode == 0 and last:
        good(last[-1])
    else:
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        bad("계산 시험이 실패함:\n           " + "\n           ".join(tail), "이 출력을 연구실 Claude 에 보낸다")


def check_cam(index, tracker, save):
    print(f"[4] 웹캠 {index}")
    import cv2
    import numpy as np
    from recorder import open_camera
    cap, frame = open_camera(index, 1280, 720)
    if cap is None:
        bad(f"웹캠 {index} 을 못 엶", "다른 프로그램(줌·카메라 앱)을 끄고, 맥은 설정 > 개인정보 보호 > 카메라 에서 이 프로그램을 켠다. "
            "외장 카메라면 --cam 1")
        return
    t0, n, hands = time.monotonic(), 0, 0
    while n < 45:
        ok, f = cap.read()
        if not ok:
            break
        frame, n = f, n + 1
        if tracker is not None and n % 3 == 0 and tracker.detect(frame, 1000 + n * 33):
            hands += 1
    fps = n / max(time.monotonic() - t0, 1e-3)
    cap.release()
    h, w = frame.shape[:2]
    bright = float(frame.mean())
    cv2.imwrite(save, frame)
    good(f"화면 {w}x{h}, 초당 {fps:.0f}장, 밝기 {bright:.0f}/255, 사진 → {os.path.basename(save)}")
    if (w, h) != (1280, 720):
        print(f"         (1280x720 이 아님. 그래도 되지만 보정은 이 크기로 한다)")
    if bright < 10:
        bad("화면이 거의 까맘", "카메라 가림막, 맥 카메라 권한(설정 > 개인정보 보호 > 카메라)을 본다")
    elif fps < 10:
        bad(f"초당 {fps:.0f}장으로 느림", "밝은 곳으로, 다른 프로그램을 끄고, 노트북 전원을 꽂는다")
    if tracker is not None:
        print(f"         손이 보인 장면 {hands}/{n // 3} (손을 안 들었으면 0 이 맞음)")


def check_robot_pc(addr):
    print(f"[5] 연구실 PC ({addr})")
    from net import Link
    try:
        link = Link(addr)
    except OSError as e:
        bad(f"주소를 못 알아들음: {e}", "연구실 PC 의 robot_server.py 창에 나온 '이 PC 주소' 를 그대로 쓴다")
        return
    # 손 없음·따라가기 꺼짐만 보낸다. robot_server.py 는 따라가기가 꺼짐 → 켜짐으로 바뀔 때만 팔을 움직이므로 팔은 그대로다.
    pkt = dict(hand=None, follow=0, rec=0, grip=1, yawf=1, sq=0.025, disc=0)
    st, rtts, t0 = None, [], time.monotonic()
    while time.monotonic() - t0 < 4.0 and len(rtts) < 10:
        link.send(bye=0, **pkt)
        t_send = time.monotonic()
        while time.monotonic() - t_send < 0.3:              # 이 패킷의 답장을 기다림 (연구실 PC 는 초당 30번 답함)
            s = link.poll()
            if s and s.get("ack") == link.seq:
                st = s
                rtts.append(link.rtt)
                break
            time.sleep(0.002)
    for _ in range(3):
        link.send(bye=1, **pkt)
        time.sleep(0.02)
    link.close()
    if st is None:
        bad("답장이 없음", "연구실 PC 에서 robot_server.py 가 켜져 있는지, 노트북 teleop.py 가 꺼져 있는지, "
            f"ping {link.addr[0]} 가 되는지 (사용설명서 6-3)")
        return
    rtts.sort()
    good(f"답장 옴: 답장까지 가운데 {rtts[len(rtts) // 2] * 1000:.0f} ms ({len(rtts)}번), 연구실 쪽은 " + ("실물 팔 (REAL ARM)" if st.get("real")
                                                                             else "가상 팔"))


def report(calib_path, checks_path):
    import numpy as np
    v = versions()
    out = ["=== webcam_piper 노트북 결과 (연구실 Claude 에 붙여 넣기) ===",
           f"컴퓨터: {v['os']}, Python {v['python']}, mediapipe {v['mediapipe']}, opencv {v['cv2']}, numpy {v['numpy']}"]
    if not os.path.exists(calib_path):
        out.append("보정: 아직 없음 (teleop.py 에서 종이 모서리 → 손 펴서 S 까지 하면 생김)")
        print("\n".join(out + ["==="]))
        return
    with open(calib_path, encoding="utf-8") as f:
        c = json.load(f)
    t = c["table"]
    cp = np.array(t["cam_pos_table_m"]) * 100
    out.append(f"보정 시각: {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(calib_path)))}")
    out.append(f"카메라: 화면 {t['w']}x{t['h']}, 초점 {t['f']:.0f} px ({t['focal_source']}), 모서리 오차 {t['reproj_px']:.1f} px")
    out.append(f"카메라 자리 (종이 + 기준): 앞 {cp[0]:.0f}, 왼쪽 {cp[1]:.0f}, 높이 {cp[2]:.0f} cm")
    if "scale" in c:
        po = np.array(c.get("palm_off_m", [np.nan] * 3)) * 100
        mv = c.get("mirror_votes", ["?", "?"])
        out.append(f"손 보정: 배율 {c['scale']:.3f}, 손바닥 가운데가 + 에서 x {po[0]:+.1f} y {po[1]:+.1f} cm, "
                   f"손 3D {'MIRRORED' if c.get('mirror') else 'ok'} (거울상 표 {mv[0]}/{mv[1]})")
        if c.get("label_votes"):
            lv = c["label_votes"]
            out.append(f"MediaPipe 가 이 {'오른' if c.get('hand', 'right') == 'right' else '왼'}손을 부르는 이름: "
                       f"'{lv[2]}' ({lv[0]}/{lv[1]}), 두 손이 보일 때 '{c.get('label')}' 쪽을 씀")
    rows = []
    if os.path.exists(checks_path):
        with open(checks_path, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        rows = [r for r in rows if r.get("t", 0) >= os.path.getmtime(calib_path) - 1]      # 지금 보정 뒤에 잰 것만
    if not rows:
        out.append("P 위치 확인: 아직 없음 (손가락 끝 다섯을 모아 모서리·+ 에 대고 P)")
    else:
        out.append(f"P 위치 확인 {len(rows)}번 (가까운 점과의 차이, cm):")
        for r in rows:
            o, hnd = r["off_cm"], r["hand_cm"]
            out.append(f"  {r['point']:10s} x {o[0]:+5.1f}  y {o[1]:+5.1f}  높이 {hnd[2]:4.1f}")
        e = np.array([np.hypot(*r["off_cm"][:2]) for r in rows])
        verdict = "좋음" if np.median(e) <= 1.5 else ("쓸 만함" if np.median(e) <= 3.0 else "다시 보정하거나 카메라를 높여야 함")
        out.append(f"  바닥 위 차이: 가운데값 {np.median(e):.1f} cm, 가장 큰 것 {e.max():.1f} cm → {verdict}")
    print("\n".join(out + ["==="]))


def main():
    ap = argparse.ArgumentParser(description="웹캠 PiPER 노트북 점검")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--no-cam", action="store_true", help="웹캠은 보지 않음")
    ap.add_argument("--robot-pc", default=None, metavar="주소[:포트]", help="연구실 PC robot_server.py 와 주고받기만")
    ap.add_argument("--report", action="store_true", help="보정·위치 확인 결과만 보여 줌")
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    ap.add_argument("--checks", default=os.path.join(HERE, "checks.jsonl"))
    ap.add_argument("--save", default=os.path.join(HERE, "cam_check.jpg"))
    a = ap.parse_args()
    if a.report:
        report(a.calib, a.checks)
        return
    if a.robot_pc and a.no_cam:                               # 연결만 볼 때는 다른 것을 건너뜀
        check_robot_pc(a.robot_pc)
    else:
        if check_programs():
            tracker = check_model()
            check_math()
            if not a.no_cam:
                check_cam(a.cam, tracker, a.save)
            if tracker is not None:
                tracker.close()
        if a.robot_pc:
            check_robot_pc(a.robot_pc)
    print()
    print("점검: 모두 됨" if not BAD else f"점검: {len(BAD)}개 안 됨 (위 '안 됨' 줄을 본다)")
    raise SystemExit(1 if BAD else 0)


if __name__ == "__main__":
    from autoupdate import update_and_restart
    update_and_restart()                                    # 켤 때 GitHub 에서 새 코드를 받아 옴 (11/17)
    main()
