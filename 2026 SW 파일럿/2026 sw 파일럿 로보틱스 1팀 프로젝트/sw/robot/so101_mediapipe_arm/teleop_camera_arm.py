#!/usr/bin/env python3
"""MediaPipe Pose + Hands로 SO-101 follower를 상대 관절 제어한다."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

from arm_mapper import ALL_JOINTS, JointLimit, RelativeArmMapper, extract_human_arm


HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = (
    "~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json"
)
DEFAULT_HARDWARE_PYTHON = "/home/user/miniconda3/envs/lerobot312/bin/python"


class HardwareClient:
    """Feetech SDK 환경의 JSON-lines bridge를 동기식으로 호출한다."""

    def __init__(self, python_path: str, port: str, calibration_path: str):
        command = [
            python_path,
            "-u",
            str(HERE / "so101_hardware_bridge.py"),
            "--port",
            port,
            "--calibration",
            str(Path(calibration_path).expanduser()),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )

    def request(self, kind: str, **payload):
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("하드웨어 브리지가 시작되지 않았습니다.")
        message = {"type": kind, **payload}
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"하드웨어 브리지가 종료되었습니다 (code={self.process.poll()}).")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "알 수 없는 모터 통신 오류"))
        return response

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request("close")
        except (BrokenPipeError, RuntimeError):
            pass
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {"camera", "person_side", "control_hz", "smoothing", "gripper", "joints"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config.json에 설정이 없습니다: {sorted(missing)}")
    if set(config["joints"]) != set(ALL_JOINTS[:-1]):
        raise ValueError("config.json joints에는 body 5개 관절만 정확히 있어야 합니다.")
    return config


def load_limits(calibration_path: Path) -> dict[str, JointLimit]:
    with calibration_path.expanduser().open(encoding="utf-8") as file:
        data = json.load(file)
    missing = set(ALL_JOINTS) - set(data)
    if missing:
        raise ValueError(f"보정 파일에 SO-101 관절이 없습니다: {sorted(missing)}")
    return {
        joint: JointLimit(int(data[joint]["range_min"]), int(data[joint]["range_max"]))
        for joint in ALL_JOINTS
    }


def midpoint_positions(limits: dict[str, JointLimit]) -> dict[str, int]:
    return {joint: round((limit.range_min + limit.range_max) / 2) for joint, limit in limits.items()}


def put_lines(frame, lines: list[str], color=(255, 255, 255)) -> None:
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        y += 24


def draw_targets(frame, targets: dict[str, int] | None) -> None:
    if not targets:
        return
    y = frame.shape[0] - 150
    for joint in ALL_JOINTS:
        text = f"{joint:14s} {targets[joint]:4d}"
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (80, 255, 255), 1, cv2.LINE_AA)
        y += 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="V4L2 카메라 번호 (기본: 0)")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--port", default="/dev/ttyACM1", help="SO-101 follower 포트")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="SO-101 follower 보정 JSON")
    parser.add_argument("--hardware-python", default=DEFAULT_HARDWARE_PYTHON)
    parser.add_argument(
        "--enable-arm",
        action="store_true",
        help="명시적으로 줄 때만 Feetech 모터 포트를 열고 실제 제어를 허용",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    calibration_path = Path(args.calibration).expanduser()
    limits = load_limits(calibration_path)
    mapper = RelativeArmMapper(limits, config)

    hardware = None
    arm_positions = midpoint_positions(limits)
    status = "DRY RUN: USB 모터 제어 없음"
    if args.enable_arm:
        if not Path(args.hardware_python).is_file():
            raise FileNotFoundError(f"LeRobot Python을 찾을 수 없습니다: {args.hardware_python}")
        hardware = HardwareClient(args.hardware_python, args.port, str(calibration_path))
        response = hardware.request("connect")
        arm_positions = {joint: int(value) for joint, value in response["positions"].items()}
        status = f"CONNECTED {args.port}: C로 사람 팔 기준 설정"

    camera_cfg = config["camera"]
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
    capture.set(cv2.CAP_PROP_FPS, int(camera_cfg["fps"]))
    if not capture.isOpened():
        if hardware is not None:
            hardware.close()
        raise RuntimeError(f"카메라 /dev/video{args.camera}를 열 수 없습니다.")

    drawing = mp.solutions.drawing_utils
    pose_module = mp.solutions.pose
    hands_module = mp.solutions.hands
    tracking = False
    armed_once = False
    last_send = 0.0
    targets = None
    last_human = None
    try:
        with pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        ) as pose, hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55,
        ) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    status = "카메라 프레임을 읽지 못함: 추종 정지"
                    tracking = False
                    time.sleep(0.05)
                    continue
                if bool(camera_cfg.get("mirror", True)):
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb)
                hand_result = hands.process(rgb)
                last_human = extract_human_arm(
                    pose_result.pose_landmarks,
                    hand_result.multi_hand_landmarks,
                    hand_result.multi_handedness,
                    config["person_side"],
                )

                if pose_result.pose_landmarks:
                    drawing.draw_landmarks(frame, pose_result.pose_landmarks, pose_module.POSE_CONNECTIONS)
                if hand_result.multi_hand_landmarks:
                    for hand in hand_result.multi_hand_landmarks:
                        drawing.draw_landmarks(frame, hand, hands_module.HAND_CONNECTIONS)

                if tracking and last_human is None:
                    tracking = False
                    status = "손/팔 추적 유실: 일시정지 (C 후 Space로 재개)"
                if last_human is not None:
                    targets = mapper.targets(last_human)
                    if tracking and targets is not None:
                        now = time.monotonic()
                        if now - last_send >= 1.0 / float(config["control_hz"]):
                            if hardware is not None:
                                hardware.request("targets", targets=targets)
                            last_send = now

                mode = "TRACKING" if tracking else "PAUSED"
                mode_color = (40, 230, 40) if tracking else (20, 210, 255)
                calibration_state = "CALIBRATED" if mapper.calibrated else "PRESS C TO CALIBRATE"
                put_lines(
                    frame,
                    [
                        f"{mode} | {calibration_state}",
                        status,
                        "C: human zero | SPACE: start/pause | R: reset arm zero | X: torque off | Q/ESC: quit",
                    ],
                    mode_color,
                )
                draw_targets(frame, targets)
                cv2.imshow("SO-101 MediaPipe arm control", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if key in (ord("c"), ord("C")):
                    if last_human is None:
                        status = "기준 설정 실패: 화면에 선택한 팔과 손을 모두 보이게 하세요"
                    elif tracking:
                        status = "추종 중에는 기준을 바꿀 수 없습니다. Space로 먼저 일시정지하세요"
                    else:
                        mapper.calibrate(last_human, arm_positions)
                        targets = mapper.targets(last_human)
                        status = "CALIBRATED: Space를 누르기 전까지 모터는 움직이지 않습니다"
                elif key == ord(" "):
                    if not mapper.calibrated:
                        status = "먼저 C로 사람 팔의 기준 자세를 저장하세요"
                    elif last_human is None:
                        status = "선택한 팔과 손이 검출될 때만 추종을 시작할 수 있습니다"
                    else:
                        tracking = not tracking
                        if tracking:
                            if hardware is not None and not armed_once:
                                hardware.request("enable")
                                armed_once = True
                            status = "TRACKING: 천천히 한 관절씩 움직여 방향을 확인하세요"
                        else:
                            if hardware is not None:
                                hardware.request("freeze")
                            status = "PAUSED: 마지막 자세 유지, 새 목표 전송 없음"
                elif key in (ord("r"), ord("R")):
                    if tracking:
                        status = "추종 중에는 R을 쓸 수 없습니다. Space로 일시정지하세요"
                    elif last_human is None:
                        status = "선택한 팔과 손을 검출한 뒤 R을 누르세요"
                    else:
                        if hardware is not None:
                            response = hardware.request("positions")
                            arm_positions = {joint: int(value) for joint, value in response["positions"].items()}
                        mapper.calibrate(last_human, arm_positions)
                        targets = mapper.targets(last_human)
                        status = "로봇 현재 자세와 사람 현재 자세를 새 기준으로 저장했습니다"
                elif key in (ord("x"), ord("X")):
                    tracking = False
                    if hardware is not None:
                        hardware.request("disable")
                        status = "TORQUE OFF: 팔을 반드시 지지하세요. 재시작하면 다시 enable 됩니다"
                    else:
                        status = "DRY RUN: 토크를 끌 실제 모터 연결이 없습니다"
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if hardware is not None:
            hardware.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
