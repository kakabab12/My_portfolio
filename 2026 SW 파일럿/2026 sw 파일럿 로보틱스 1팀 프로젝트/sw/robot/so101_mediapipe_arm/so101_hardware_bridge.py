#!/usr/bin/env python3
"""SO-101 Feetech 모터 전용 JSON-lines 브리지.

표준 출력은 부모 프로그램이 읽는 JSON 응답만 기록한다. 진단 로그는 표준 오류로
보내므로, 이 프로그램은 독립 실행 대신 `teleop_camera_arm.py`에서 시작한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


JOINT_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
GOAL_VELOCITY = 46
PRESENT_POSITION = 56


class SO101Hardware:
    def __init__(self, port: str, calibration_path: Path):
        try:
            import scservo_sdk as scs
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "feetech-servo-sdk가 필요합니다. LeRobot 환경으로 실행해야 합니다."
            ) from error
        self.scs = scs
        self.port = port
        self.calibration = self._load_calibration(calibration_path)
        self.port_handler = scs.PortHandler(port)
        self.packet_handler = scs.PacketHandler(0)  # SO-101 STS3215 = Feetech protocol 0
        self.connected = False

    @staticmethod
    def _load_calibration(path: Path) -> dict[str, dict[str, int]]:
        if not path.is_file():
            raise FileNotFoundError(f"보정 파일을 찾을 수 없습니다: {path}")
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        missing = set(JOINT_IDS) - set(data)
        if missing:
            raise ValueError(f"보정 파일에 관절이 없습니다: {sorted(missing)}")
        return data

    def connect(self) -> dict[str, int]:
        if not self.port_handler.openPort():
            raise ConnectionError(f"포트를 열 수 없습니다: {self.port}")
        if not self.port_handler.setBaudRate(1_000_000):
            self.port_handler.closePort()
            raise ConnectionError(f"1 Mbps로 설정할 수 없습니다: {self.port}")
        self.connected = True
        try:
            # 토크를 바꾸지 않고 모든 모터의 현재 위치만 읽어 연결을 검증한다.
            return self.read_positions()
        except Exception:
            self.close()
            raise

    def read_positions(self) -> dict[str, int]:
        self._require_connection()
        result = {}
        for joint, motor_id in JOINT_IDS.items():
            value, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, PRESENT_POSITION
            )
            self._raise_on_status(joint, comm_result, error)
            result[joint] = int(value)
        return result

    def enable_torque(self) -> None:
        self._require_connection()
        for joint, motor_id in JOINT_IDS.items():
            # 보수적인 가속/속도로 시작한다. SRAM 값만 바꾸므로 전원을 끄면 원상복구된다.
            self._write_1byte(joint, motor_id, ACCELERATION, 20)
            self._write_2byte(joint, motor_id, GOAL_VELOCITY, 120)
            self._write_1byte(joint, motor_id, TORQUE_ENABLE, 1)

    def disable_torque(self) -> None:
        self._require_connection()
        for joint, motor_id in JOINT_IDS.items():
            self._write_1byte(joint, motor_id, TORQUE_ENABLE, 0)

    def write_targets(self, targets: dict[str, int]) -> None:
        self._require_connection()
        if set(targets) != set(JOINT_IDS):
            raise ValueError("6개 관절 목표값을 모두 보내야 합니다.")
        writer = self.scs.GroupSyncWrite(
            self.port_handler, self.packet_handler, GOAL_POSITION, 2
        )
        for joint, motor_id in JOINT_IDS.items():
            bounds = self.calibration[joint]
            target = int(targets[joint])
            target = max(int(bounds["range_min"]), min(int(bounds["range_max"]), target))
            data = [
                self.scs.SCS_LOBYTE(self.scs.SCS_LOWORD(target)),
                self.scs.SCS_HIBYTE(self.scs.SCS_LOWORD(target)),
            ]
            if not writer.addParam(motor_id, data):
                raise RuntimeError(f"{joint} 목표값을 sync write에 넣지 못했습니다.")
        comm_result = writer.txPacket()
        writer.clearParam()
        if comm_result != self.scs.COMM_SUCCESS:
            raise ConnectionError(self.packet_handler.getTxRxResult(comm_result))

    def close(self) -> None:
        if self.connected:
            self.port_handler.closePort()
            self.connected = False

    def _write_1byte(self, joint: str, motor_id: int, address: int, value: int) -> None:
        comm_result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        self._raise_on_status(joint, comm_result, error)

    def _write_2byte(self, joint: str, motor_id: int, address: int, value: int) -> None:
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, address, value
        )
        self._raise_on_status(joint, comm_result, error)

    def _raise_on_status(self, joint: str, comm_result: int, error: int) -> None:
        if comm_result != self.scs.COMM_SUCCESS:
            raise ConnectionError(f"{joint}: {self.packet_handler.getTxRxResult(comm_result)}")
        if error:
            raise ConnectionError(f"{joint}: {self.packet_handler.getRxPacketError(error)}")

    def _require_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("SO-101 포트가 연결되지 않았습니다.")


def reply(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()

    arm = None
    try:
        arm = SO101Hardware(args.port, Path(args.calibration).expanduser())
        for raw_line in sys.stdin:
            try:
                message = json.loads(raw_line)
                kind = message["type"]
                if kind == "connect":
                    payload = {"ok": True, "positions": arm.connect()}
                elif kind == "positions":
                    payload = {"ok": True, "positions": arm.read_positions()}
                elif kind == "enable":
                    arm.enable_torque()
                    payload = {"ok": True}
                elif kind == "freeze":
                    # 마지막 Goal_Position을 유지한다. 토크는 그대로 둔다.
                    payload = {"ok": True}
                elif kind == "disable":
                    arm.disable_torque()
                    payload = {"ok": True}
                elif kind == "targets":
                    arm.write_targets(message["targets"])
                    payload = {"ok": True}
                elif kind == "close":
                    payload = {"ok": True}
                    reply(payload)
                    break
                else:
                    raise ValueError(f"알 수 없는 명령: {kind}")
            except Exception as error:  # 부모가 화면에 표시할 수 있도록 오류를 JSON으로 반환
                payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            reply(payload)
    finally:
        if arm is not None:
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
