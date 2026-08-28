"""MediaPipe(x, y) 팔 추적으로 SO-ARM-101 을 원격 조종한다.

  python teleop_so101.py --port COM5              # 실제 로봇 구동
  python teleop_so101.py --dry-run                # 로봇 없이 값만 확인
  python teleop_so101.py --port COM5 --side left  # 왼팔 추적

자연스러운 동작을 위해 두 단계로 필터링한다:
  1) 1€ Filter (Casiez et al., CHI 2012) 로 랜드마크 잡음을 속도 적응형으로 제거
  2) critically-damped SmoothDamp 로 로봇 명령을 가속/감속 곡선으로 부드럽게 이동

ESC 또는 q 로 종료. 종료 시 로봇은 안전하게 disconnect 된다.
"""

import argparse
import json
import os
import time

import cv2

from arm_tracker import JOINT_LIMITS, ArmTracker

TUNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.json")


def load_tuning(tracker, path=TUNING_FILE):
    """저장된 0점/반전 설정을 tracker 에 적용한다. 없으면 조용히 넘어간다."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"튜닝 파일을 읽지 못했습니다 ({exc}). 기본값으로 시작합니다.")
        return False

    for name, val in data.get("offsets", {}).items():
        if name in tracker.offsets:
            tracker.offsets[name] = float(val)
    inverts = data.get("invert", [])
    tracker.invert_joints = {n for n in inverts if n in JOINT_LIMITS}
    if "gain" in data:
        tracker.gain = float(data["gain"])
    for name, val in data.get("joint_gains", {}).items():
        if name in tracker.joint_gains:
            tracker.joint_gains[name] = float(val)
    return True


def save_tuning(tracker, path=TUNING_FILE):
    """현재 0점/반전 설정을 파일로 저장한다."""
    data = {
        "offsets": {n: round(v, 1) for n, v in tracker.offsets.items() if v},
        "invert": sorted(tracker.invert_joints),
        "gain": round(tracker.gain, 2),
        "joint_gains": {n: round(v, 2)
                        for n, v in tracker.joint_gains.items() if v != 1.0},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # 쓰다가 중단돼도 기존 파일이 깨지지 않게
    return data


def smooth_damp(current, target, velocity, smooth_time, dt, max_speed):
    """Critically-damped spring — Game Programming Gems 4 / Unity Mathf.SmoothDamp.

    선형 램프(고정 각도씩 이동)와 달리 가속·감속이 이어지는 자연스러운
    곡선을 그린다. smooth_time 이 반응 속도, max_speed(도/초) 가 안전 상한이다.
    """
    smooth_time = max(smooth_time, 1e-4)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp_term = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

    change = current - target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    clamped_target = current - change

    temp = (velocity + omega * change) * dt
    new_velocity = (velocity - omega * temp) * exp_term
    new_value = clamped_target + (change + temp) * exp_term

    # 목표를 지나쳐 진동하는 것 방지
    if (target - current > 0.0) == (new_value > target):
        new_value = target
        new_velocity = (new_value - target) / dt if dt > 0 else 0.0
    return new_value, new_velocity


def build_robot(port, robot_id):
    """lerobot 의 SO101Follower 를 연결해서 반환."""
    try:
        # lerobot >= 0.4
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="팔로워 시리얼 포트 (예: COM5)")
    ap.add_argument("--id", default="so101", help="lerobot 캘리브레이션 id")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--view", default="front", choices=["front", "side"],
                    help="카메라 위치. side = 측면 촬영(좌우 pan 은 관측 불가)")
    ap.add_argument("--pan", type=float, default=0.0,
                    help="shoulder_pan 을 고정할 각도 (--view side 또는 --lock-pan 일 때 적용)")
    ap.add_argument("--pan-mode", default="depth",
                    choices=["depth", "azimuth", "lateral"],
                    help="shoulder_pan 계산 방식. "
                         "depth=팔이 가까우면 왼쪽/멀면 오른쪽, "
                         "azimuth=수평면 방위각(0도 정면), "
                         "lateral=화면상 좌우 오프셋(z 미사용)")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="z 값 배율. z 는 부정확하므로 좌우 반응이 약하면 올리고 "
                         "과하면 낮춘다")
    ap.add_argument("--pan-gain", type=float, default=1.0,
                    help="shoulder_pan 각도 배율")
    ap.add_argument("--depth-span", type=float, default=4.0,
                    help="depth 모드에서 중립 대비 이 만큼 변하면 좌우 최대(90도). "
                         "작을수록 민감")
    ap.add_argument("--pan-cutoff", type=float, default=0.5,
                    help="shoulder_pan 전용 1€ 필터 컷오프(Hz). depth 노이즈가 크므로 "
                         "다른 축보다 훨씬 낮게 잡는다. 떨리면 더 낮추고 굼뜨면 올린다")
    ap.add_argument("--pan-beta", type=float, default=0.0,
                    help="shoulder_pan 전용 1€ 필터 속도 계수")
    ap.add_argument("--pan-deadzone", type=float, default=0.35,
                    help="팔이 이 비율(위팔길이 대비)보다 덜 뻗으면 pan 을 갱신하지 않는다")
    ap.add_argument("--lock-pan", action="store_true",
                    help="정면 촬영에서도 좌우 움직임을 무시하고 shoulder_pan 을 --pan 값에 고정")
    ap.add_argument("--elbow-gain", type=float, default=1.0,
                    help="팔꿈치 각도 배율. 1.0 = 실제 팔꿈치 각도를 1:1 로 반영")
    ap.add_argument("--elbow-offset", type=float, default=-100.0,
                    help="팔꿈치 0점 보정(도). 로봇의 0도는 중간 위치라, 사람이 팔을 "
                         "쭉 폈을 때(각도 0) 로봇도 펴지도록 -100 만큼 내려준다")
    ap.add_argument("--lift-gain", type=float, default=1.0,
                    help="어깨 각도 배율. 1.0 = 실제 어깨 각도를 1:1 로 반영")
    ap.add_argument("--wrist-gain", type=float, default=1.0,
                    help="손목 각도 배율. 1.0 = 실제 손목 각도를 1:1 로 반영")
    ap.add_argument("--track-roll", action="store_true",
                    help="wrist_roll 을 손 회전으로 추적한다 (기본은 고정)")
    ap.add_argument("--roll-hold", type=float, default=0.0,
                    help="wrist_roll 고정 각도")
    ap.add_argument("--roll-neutral", type=float, default=90.0,
                    help="wrist_roll 중립 오프셋(도). 손을 편하게 폈을 때 0 이 되도록 맞춘다")
    ap.add_argument("--invert", default="shoulder_lift,shoulder_pan",
                    help="방향이 반대로 나오는 관절을 콤마로 나열 (예: elbow_flex,gripper). "
                         "빈 문자열이면 전부 정방향")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gain", type=float, default=1.8,
                    help="전체 동작 배율. 1.0=사람 각도와 1:1, 클수록 작은 동작에도 크게 반응")
    ap.add_argument("--joint-gain", default="shoulder_lift=0.68",
                    help="관절별 배율. 전체 배율에 곱해진다. "
                         "예: shoulder_lift=2.0,elbow_flex=1.2")
    ap.add_argument("--beta", type=float, default=0.6,
                    help="1€ 필터 속도 계수. 클수록 빠른 동작의 지연이 줄지만 떨림은 늘어남")
    ap.add_argument("--min-cutoff", type=float, default=2.5,
                    help="1€ 필터 최소 컷오프(Hz). 작을수록 정지 시 부드럽지만 "
                         "작고 느린 동작이 눌려서 안 따라온다")
    ap.add_argument("--smooth-time", type=float, default=0.22,
                    help="로봇 명령 ease-in/out 시간(초). 클수록 부드럽지만 반응이 느려짐")
    ap.add_argument("--max-speed", type=float, default=90.0,
                    help="관절당 최대 각속도(도/초) 안전 상한. "
                         "배율(--gain)과 독립이라 '많이, 하지만 천천히' 가 가능하다")
    ap.add_argument("--no-tuning", action="store_true",
                    help="저장된 tuning.json 을 무시하고 명령줄 기본값으로 시작")
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 화면만")
    args = ap.parse_args()

    if not args.dry_run and not args.port:
        ap.error("--port 를 주거나 --dry-run 을 사용하세요.")

    robot = None if args.dry_run else build_robot(args.port, args.id)
    invert_joints = [j.strip() for j in args.invert.split(",") if j.strip()]
    tracker = ArmTracker(args.side, beta=args.beta, min_cutoff=args.min_cutoff,
                         camera_index=args.camera,
                         view=args.view, pan_hold=args.pan, lock_pan=args.lock_pan,
                         invert_joints=invert_joints,
                         wrist_gain=args.wrist_gain, roll_neutral=args.roll_neutral,
                         lift_gain=args.lift_gain,
                         elbow_gain=args.elbow_gain, elbow_offset=args.elbow_offset,
                         lock_roll=not args.track_roll, roll_hold=args.roll_hold,
                         gain=args.gain,
                         pan_mode=args.pan_mode, depth_scale=args.depth_scale,
                         pan_gain=args.pan_gain, pan_deadzone=args.pan_deadzone,
                         depth_span=args.depth_span,
                         pan_cutoff=args.pan_cutoff, pan_beta=args.pan_beta)
    period = 1.0 / args.fps

    # 급출발 방지: 첫 명령은 로봇의 현재 자세에서 출발한다.
    cmd = {k: 0.0 for k in JOINT_LIMITS}
    if robot is not None:
        obs = robot.get_observation()
        for k in cmd:
            cmd[k] = float(obs.get(f"{k}.pos", 0.0))
    vel = {k: 0.0 for k in JOINT_LIMITS}

    joint_names = list(JOINT_LIMITS)
    tracker.offsets["elbow_flex"] += args.elbow_offset
    sel = joint_names.index("shoulder_lift")  # 처음 선택된 관절

    for item in args.joint_gain.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, val = item.partition("=")
        name = name.strip()
        if name not in JOINT_LIMITS:
            ap.error(f"--joint-gain 에 알 수 없는 관절: {name}")
        try:
            tracker.joint_gains[name] = float(val)
        except ValueError:
            ap.error(f"--joint-gain 값이 숫자가 아닙니다: {item}")

    # 저장된 튜닝이 있으면 덮어쓴다 (명령줄 기본값보다 우선).
    if not args.no_tuning and load_tuning(tracker):
        print(f"저장된 튜닝을 불러왔습니다: {os.path.basename(TUNING_FILE)}")

    print("=" * 62)
    print("추적 시작.  ESC 또는 q = 종료")
    print("  1~6      관절 선택: " + " ".join(
        f"{i + 1}={n}" for i, n in enumerate(joint_names)))
    print("  [ / ]    선택한 관절의 0점을 -5 / +5 도 조정")
    print("  - / =    전체 동작 배율을 -0.2 / +0.2 조정")
    print("  , / .    선택한 관절만 배율을 -0.2 / +0.2 조정")
    print("  c        현재 자세를 깊이 중립(pan 0도)으로 재설정")
    print("  i        선택한 관절의 방향 뒤집기")
    print("  w        현재 설정을 파일에 저장 (다음 실행부터 자동 적용)")
    print("  p        현재 설정을 화면에 출력")
    print("  s        스냅샷 저장")
    print(f"선택됨: {joint_names[sel]}   반전: {sorted(tracker.invert_joints) or '없음'}")
    print("=" * 62)
    prev_t = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            dt = max(t0 - prev_t, 1e-3)
            prev_t = t0

            action, frame = tracker.read()
            if frame is None:
                break

            if action is not None:
                # 목표값까지 critically-damped 곡선으로 부드럽게 따라간다.
                for k, target in action.items():
                    cmd[k], vel[k] = smooth_damp(
                        cmd[k], target, vel[k], args.smooth_time, dt, args.max_speed
                    )
                if robot is not None:
                    robot.send_action({f"{k}.pos": v for k, v in cmd.items()})

            cv2.imshow("SO-ARM-101 teleop (x,y only)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                cv2.imwrite("debug_snapshot.jpg", frame)
                print("스냅샷 저장: debug_snapshot.jpg  action=",
                      {k: round(v, 1) for k, v in cmd.items()})
            if ord("1") <= key <= ord("6"):
                sel = key - ord("1")
                name = joint_names[sel]
                print(f"선택: {name}  (0점 {tracker.offsets[name]:+.0f}도, "
                      f"반전 {'ON' if name in tracker.invert_joints else 'OFF'})")
            if key in (ord("["), ord("]")):
                name = joint_names[sel]
                tracker.offsets[name] += -5.0 if key == ord("[") else 5.0
                tracker.filters.reset(name)
                print(f"[{name}] 0점 = {tracker.offsets[name]:+.0f}도")
            if key in (ord("-"), ord("=")):
                tracker.gain = max(0.2, tracker.gain + (-0.2 if key == ord("-") else 0.2))
                print(f"전체 배율 = {tracker.gain:.1f}")
            if key in (ord(","), ord(".")):
                name = joint_names[sel]
                cur = tracker.joint_gains.get(name, 1.0)
                tracker.joint_gains[name] = max(
                    0.2, cur + (-0.2 if key == ord(",") else 0.2))
                print(f"[{name}] 개별 배율 = {tracker.joint_gains[name]:.1f}"
                      f"  (실효 배율 {tracker.gain * tracker.joint_gains[name]:.1f})")
            if key == ord("c"):
                tracker.reset_depth_reference()
                tracker.filters.reset("shoulder_pan")
                print("현재 자세를 깊이 중립(pan 0도)으로 다시 잡습니다...")
            if key == ord("i"):
                name = joint_names[sel]
                if name in tracker.invert_joints:
                    tracker.invert_joints.discard(name)
                else:
                    tracker.invert_joints.add(name)
                # 필터 이력을 비워 반전 직후 값이 튀지 않게 한다.
                tracker.filters.reset(name)
                print(f"[{name}] 반전 {'ON' if name in tracker.invert_joints else 'OFF'}")
            if key == ord("w"):
                data = save_tuning(tracker)
                print(f"저장 완료 -> {os.path.basename(TUNING_FILE)}  {data}")
            if key == ord("p"):
                offs = "  ".join(f"{n}={v:+.0f}"
                                 for n, v in tracker.offsets.items() if v)
                jg = "  ".join(f"{n}={v:.1f}"
                               for n, v in tracker.joint_gains.items() if v != 1.0)
                print("---- 현재 설정 (저장하려면 w) ----")
                print(f"  배율: 전체 {tracker.gain:.1f}   관절별 {jg or '없음'}")
                print(f"  반전: {sorted(tracker.invert_joints) or '없음'}")
                print(f"  0점 : {offs or '없음'}")

            rest = period - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cv2.destroyAllWindows()
        if robot is not None:
            robot.disconnect()
        print("종료했습니다.")


if __name__ == "__main__":
    main()
