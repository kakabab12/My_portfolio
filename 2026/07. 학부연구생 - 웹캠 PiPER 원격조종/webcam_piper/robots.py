"""팔 연결 — 미리보기(가상 팔) / agx-bus(연구실 표준) / piper_sdk (2026-11-15).

모두 같은 꼴:  joints() 관절 6개 (rad)   gripper() 집게 폭 (m)
               send(q, width)  관절 목표와 집게 폭      hold() 지금 자리에 멈춤      close()

실물 기록 (2026-11-12, 사용자 PC 옆 PiPER): 펌웨어 S-V1.9-0 → agx-bus 는 version 'v189' 로 붙어야 명령이 먹힌다
('auto' 가 펌웨어를 읽어 고른다). 관절 6개 10° 왕복은 됐고, 그리퍼는 명령에 반응이 없었다 (원인 아직 모름).
"""
import time

import numpy as np

from piper_kin import HOME_Q

GRIP_MAX = 0.07           # 시뮬 그리퍼와 같은 최대 폭 (m). 실물 그리퍼 설정은 0.1 m 형 (11/12)


class Preview:
    """팔 없이 화면에서만 보는 가상 팔. 보낸 관절 목표를 한 박자씩 따라간다."""
    is_real = False

    def __init__(self, q0=HOME_Q, follow=0.5):
        self.q = np.asarray(q0, np.float64).copy()
        self.target = self.q.copy()
        self.w = GRIP_MAX
        self.w_target = GRIP_MAX
        self.follow = follow

    def joints(self):
        self.q += self.follow * (self.target - self.q)
        return self.q.copy()

    def gripper(self):
        self.w += self.follow * (self.w_target - self.w)
        return self.w

    def send(self, q, width):
        self.target = np.asarray(q, np.float64).copy()
        self.w_target = float(np.clip(width, 0.0, GRIP_MAX))

    def move_joints(self, q):
        """관절만 (그리퍼는 그대로) — 처음 자세로 돌아갈 때 (11/20)."""
        self.target = np.asarray(q, np.float64).copy()

    def hold(self):
        self.target = self.q.copy()

    def close(self):
        pass


class AgxBusArm:
    """연구실 표준 agx-bus (찬혁님 piper-teleop 의 .venv 에 깔려 있음)."""
    is_real = True

    def __init__(self, can="can0", speed=30, version="auto"):
        from agx_bus import AgilexBus, AgilexGripper
        if version == "auto":
            from agx_bus.pyAgxArm import resolve_firmware_profile
            probe = AgilexBus(can, name="piper", version="default")
            probe.connect()
            fw = probe.robot.get_firmware(timeout=3.0)["software_version"]
            probe.disconnect()
            version = resolve_firmware_profile("piper", fw)
            print(f"[팔] 펌웨어 {fw} → agx-bus '{version}'")
        self.bus = AgilexBus(can, name="piper", version=version)
        self.bus.connect()
        self.bus.robot.set_speed_percent(percent=int(speed))      # connect 가 100% 로 올려 두므로 바로 낮춘다
        self.bus.enable_torque()
        self.ee = AgilexGripper(self.bus)
        self._w_sent = None

    def joints(self):
        return np.array(self.bus.read_joints().position, np.float64)

    def gripper(self):
        try:
            return float(self.ee.read_gripper().width)
        except Exception:  # noqa: BLE001 — 그리퍼를 못 읽어도 팔은 계속
            return float("nan")

    def send(self, q, width):
        self.bus.move_joints([float(x) for x in q])
        w = float(np.clip(width, 0.0, GRIP_MAX))
        if self._w_sent is None or abs(w - self._w_sent) > 0.002:   # 2 mm 넘게 바뀔 때만 보낸다
            try:
                self.ee.move_gripper(width=w)
                self._w_sent = w
            except Exception as e:  # noqa: BLE001
                print(f"[팔] 그리퍼 명령 실패: {e}")
                self._w_sent = w

    def move_joints(self, q):
        """관절만 (그리퍼는 그대로) — 처음 자세로 돌아갈 때 (11/20)."""
        self.bus.move_joints([float(x) for x in q])

    def hold(self):
        self.bus.move_joints(self.bus.read_joints().position)

    def estop(self):
        self.bus.e_stop()

    def close(self):
        try:
            self.hold()
        finally:
            self.bus.disconnect()           # 힘은 켠 채 둔다 (끄면 팔이 떨어짐 — 끌 때는 손으로 받치고 piper-ui 로)


class SdkArm:
    """piper_sdk (real_piper/robot_io.PiperSDK 와 같은 단위). 집게 폭은 이어지는 값으로 보낸다."""
    is_real = True
    RAD2MDEG = 180.0 / np.pi * 1000.0

    def __init__(self, can="can0", speed=30, interface="v2", effort_nm=1.0):
        import piper_sdk
        name = {"v2": "C_PiperInterface_V2", "v3": "C_PiperInterface_V3"}[interface]
        self.p = getattr(piper_sdk, name)(can_name=can)
        self.speed = int(np.clip(speed, 1, 100))
        self.effort = effort_nm
        self.p.ConnectPort()
        time.sleep(0.1)
        t0 = time.time()
        while not self.p.EnablePiper():
            if time.time() - t0 > 5.0:
                raise RuntimeError("팔을 켜지 못함 (EnablePiper) — CAN·전원 확인")
            time.sleep(0.01)
        self._w_sent = None

    def joints(self):
        js = self.p.GetArmJointMsgs().joint_state
        return np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
                        np.float64) / self.RAD2MDEG

    def gripper(self):
        return float(self.p.GetArmGripperMsgs().gripper_state.grippers_angle) / 1e6

    def send(self, q, width):
        self.p.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        self.p.JointCtrl(*[int(round(x * self.RAD2MDEG)) for x in q])
        w = float(np.clip(width, 0.0, GRIP_MAX))
        if self._w_sent is None or abs(w - self._w_sent) > 0.002:
            self.p.GripperCtrl(int(round(w * 1e6)), int(round(self.effort * 1000)), 0x01, 0)
            self._w_sent = w

    def move_joints(self, q):
        """관절만 (그리퍼는 그대로) — 처음 자세로 돌아갈 때 (11/20)."""
        self.p.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        self.p.JointCtrl(*[int(round(x * self.RAD2MDEG)) for x in q])

    def hold(self):
        self.send(self.joints(), self._w_sent if self._w_sent is not None else GRIP_MAX)

    def estop(self):
        self.p.EmergencyStop(0x01)

    def close(self):
        pass


def make_arm(kind, can="can0", speed=30):
    if kind == "preview":
        return Preview()
    if kind == "agxbus":
        return AgxBusArm(can, speed)
    if kind == "sdk":
        return SdkArm(can, speed)
    raise ValueError(kind)
