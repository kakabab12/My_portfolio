"""카메라·손 없이 계산 길을 시험한다 (2026-11-15, 11/16 손 전체로 쥐기·손 펴서 보정으로 바꿈).

가상 노트북 카메라(1280×720, 가로 시야 70°, 책상 위 25 cm, 종이 가운데에서 65 cm 앞)와 가상 손을 만든다.
손 21점을 그 카메라로 찍은 화면 좌표에 잡음을 넣고, MediaPipe 처럼 '손 크기를 10% 크게 아는' 3D 좌표를 준다.
그다음 teleop.py 와 같은 계산(모서리 → 카메라 자세, 손 펴서 올려놓기 → 거리 배율, 손 → 책상 좌표)으로 값을 되찾아 오차를 잰다.
마지막으로 사람이 손을 펴고 → 물체로 가서 손 전체로 쥐고 → 들고 → 옮기고 → 놓을 때 가상 팔 집게가 같은 길을 가는지 본다.

실행: python test_geometry.py   (mediapipe·웹캠·mujoco 필요 없음, opencv 필요)
맨 끝 줄이 "결과: 모두 통과 (n개)" 면 이 컴퓨터에서 계산이 제대로 도는 것이다 (11/16 더함, 노트북 Claude 가 판단할 수 있게).
6~8 은 11/16 검토에서 더함: 엉뚱한 첫 장면, MediaPipe 손 3D 모양 오차, 손 3D 거울상 판별.
"""
import math
import time

import cv2
import numpy as np

from geometry import (A4_D, A4_W, CORNERS_TABLE, FLAT_H, PINCH_H0, HandMeasure, TableCalib, focal_from_hfov, intrinsics,
                      scale_for_flat)
from mapping import Mapper
from piper_kin import TABLE_Z, PiperKin
from robots import Preview

RNG = np.random.default_rng(0)
W, H = 1280, 720
TIPS_Y = (0.022, 0.007, -0.008, -0.022)          # 검지·중지·약지·새끼 끝 (쥘 때, 손 기준 y)
CHECKS = []                                       # (무엇, 통과?) — 맨 끝에 모아서 알려 줌


def check(name, ok):
    CHECKS.append((name, bool(ok)))


def look_at(cam_pos, target):
    """책상 → 카메라 회전 R 과 이동 t (카메라 z = 앞, x = 화면 오른쪽, y = 화면 아래)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ cam_pos


def hand_model(kind, w=0.08, lift=0.0):
    """오른손 21점 (손 기준, m, MediaPipe 번호 순서).
    flat : 손바닥을 아래로 종이에 펴서 올려놓음, 손가락 앞(+x), 엄지 왼쪽(+y). 원점 = 손바닥 가운데(손목과 가운데 손가락 뿌리 사이),
           높이 = FLAT_H + lift (lift 로 '어림 높이와 실제가 다를 때'를 흉내 냄)
    grasp: 손 전체로 위에서 쥠. 엄지 끝(−w/2, 0, 0) ↔ 네 손가락 끝(+w/2, y, 0). 원점 = 두 쪽의 가운데(집기점)"""
    if kind == "flat":
        xy = [(-0.045, 0.0),
              (-0.03, 0.025), (-0.01, 0.045), (0.012, 0.06), (0.03, 0.07),
              (0.045, 0.022), (0.085, 0.026), (0.11, 0.028), (0.13, 0.03),
              (0.045, 0.0), (0.09, 0.0), (0.118, 0.0), (0.14, 0.0),
              (0.042, -0.02), (0.083, -0.022), (0.107, -0.024), (0.127, -0.026),
              (0.035, -0.038), (0.065, -0.045), (0.083, -0.049), (0.1, -0.052)]
        return np.array([[x, y, h + lift] for (x, y), h in zip(xy, FLAT_H)])
    a = w / 2
    p = [(-0.06, 0.0, 0.08),
         (-0.05, 0.02, 0.06), (-a - 0.01, 0.025, 0.04), (-a - 0.005, 0.015, 0.02), (-a, 0.0, 0.0)]
    for y, (bx, bz) in zip(TIPS_Y, ((0.0, 0.085), (0.0, 0.09), (-0.003, 0.087), (-0.008, 0.08))):
        p += [(bx, y, bz), (a + 0.01, y, 0.07), (a + 0.01, y, 0.03), (a, y, 0.0)]
    p = np.array(p)
    p[:, 1] -= np.mean(TIPS_Y) / 2                     # 집기점이 원점에 오게
    return p


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def shoot(R, t, K, pts_table, noise_px):
    img, _ = cv2.projectPoints(pts_table, cv2.Rodrigues(R)[0], t, K, None)
    return img.reshape(-1, 2) + RNG.normal(0, noise_px, (len(pts_table), 2))


def hand_obs(R, t, K, pos, yaw, kind="grasp", w=0.08, noise_px=1.0, size_err=1.10, lift=0.0):
    """손을 pos (책상 좌표) 에 yaw 만큼 돌려 둔 장면 → (MediaPipe 가 줄 손 3D, 화면 21점)."""
    local = hand_model(kind, w, lift)
    true_table = (rotz(yaw) @ local.T).T + np.asarray(pos, float)
    img = shoot(R, t, K, true_table, noise_px)
    world = local * size_err                     # MediaPipe 가 아는 손 3D: 크기 10% 틀림, 손 가운데가 원점
    world = world - world.mean(axis=0)
    return world, img


def axis_err_deg(a, b):
    return math.degrees(math.remainder(a - b, math.pi))


def main():
    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    cam_pos = np.array([0.65, 0.0, 0.25])
    R, t = look_at(cam_pos, np.array([-0.05, 0.0, 0.05]))
    print("== 1. 종이 모서리 → 카메라")
    for noise in (1.0, 3.0):
        img = shoot(R, t, K_true, CORNERS_TABLE, noise)
        c = TableCalib(img, W, H)
        print(f"  모서리 잡음 {noise:.0f}px: 초점 {c.f:.0f}px (참 {K_true[0, 0]:.0f}, {c.focal_source}), "
              f"카메라 위치 오차 {np.linalg.norm(c.cam_pos - cam_pos) * 100:.1f} cm")
        if noise == 1.0:
            check("1 종이 모서리 → 카메라 위치 3 cm 안", np.linalg.norm(c.cam_pos - cam_pos) < 0.03)
    calib = TableCalib(shoot(R, t, K_true, CORNERS_TABLE, 1.0), W, H)

    print("== 2. 손 펴서 올려놓기 → 거리 배율 (참 %.3f)" % (1 / 1.10))
    m1 = HandMeasure(calib, 1.0)
    for noise in (1.0, 3.0):
        for lift in (0.0, 0.003, -0.003):
            cams = [m1(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=noise, lift=lift))["lm_cam_raw"]
                    for _ in range(20)]
            s = scale_for_flat(calib, np.median(cams, axis=0))
            palm = np.median([HandMeasure(calib, s)(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=noise,
                                                              lift=lift))["palm"] for _ in range(20)], axis=0)
            print(f"  손 점 잡음 {noise:.0f}px, 실제 손이 어림보다 {lift * 1000:+.0f} mm: 배율 {s:.3f}, "
                  f"손바닥 가운데 오차 x {palm[0] * 100:+.1f} y {palm[1] * 100:+.1f} cm")
            if noise == 1.0 and lift == 0.0:
                check("2 손 펴서 보정한 배율 (참 0.909 ± 0.02)", abs(s - 1 / 1.10) < 0.02)

    print("== 3. 손 전체로 쥐는 손 (손 점 잡음 1·3 px, 손 크기 10% 틀림)")
    cams = [m1(*hand_obs(R, t, K_true, (0, 0, 0), 0.0, "flat", noise_px=1.0))["lm_cam_raw"] for _ in range(20)]
    meas = HandMeasure(calib, scale_for_flat(calib, np.median(cams, axis=0)))
    tests = [("+ 위, 손 벌림 8 cm", (0, 0, 0.012), 0.0, 0.08), ("물체(앞15 왼10) 쥠 5.5 cm", (0.15, 0.10, 0.03), 0.3, 0.055),
             ("들어 올림 10 cm", (0.05, -0.05, 0.12), 0.4, 0.055), ("왼쪽 뒤 5 cm 위", (-0.08, 0.12, 0.05), -0.4, 0.07),
             ("오른쪽 앞 15 cm 위", (0.10, -0.12, 0.15), 0.0, 0.10)]
    for noise in (1.0, 3.0):
        print(f"  손 점 잡음 {noise:.0f}px")
        for name, p, yaw, w in tests:
            rs = [meas(*hand_obs(R, t, K_true, p, yaw, "grasp", w, noise)) for _ in range(30)]
            e = np.array([r["pinch"] - p for r in rs])
            ye = np.array([axis_err_deg(r["yaw"], yaw - math.pi / 2) for r in rs])
            ge = np.array([r["gap"] - w for r in rs]) * 1000
            if noise == 1.0:
                check(f"3 {name}: 집기점 1 cm 안", np.linalg.norm(e.mean(0)) < 0.01)
            print(f"    {name:22s} 집기점 오차 평균 {np.linalg.norm(e.mean(0)) * 100:4.1f} cm, 흔들림 "
                  f"{np.linalg.norm(e.std(0)) * 100:4.1f} cm | 집게 방향 {ye.mean():+4.0f}°±{ye.std():.0f} | "
                  f"엄지↔손가락 {ge.mean():+3.0f}±{ge.std():.0f} mm")

    print("== 4. 매핑 + 역기구학 (가상 팔, 30 Hz): 손 펴고 → 물체로 → 쥠 → 들기 → 옮기기 → 놓기")
    kin = PiperKin()
    arm = Preview()
    mp = Mapper(kin)
    mp.sync_to(arm.joints())
    # (집기점, 손 돌림, 엄지↔손가락, 초). 손 돌림 0.5 rad = 손가락 끝이 왼쪽으로 29° 돌아감
    path = [((0, 0, 0.012), 0.0, 0.10, 2.0), ((0.15, 0.10, 0.03), 0.5, 0.09, 4.0), ((0.15, 0.10, 0.03), 0.5, 0.055, 1.0),
            ((0.15, 0.10, 0.12), 0.5, 0.055, 2.0), ((-0.05, -0.10, 0.12), 0.5, 0.055, 3.0),
            ((-0.05, -0.10, 0.03), 0.5, 0.055, 2.0), ((-0.05, -0.10, 0.03), 0.5, 0.09, 1.0)]
    dt = 1 / 30
    cur, cur_yaw = np.array([0, 0, 0.012], float), 0.0
    worst_step = 0.0
    for goal, yaw, gap, secs in path:
        goal = np.array(goal, float)
        n = int(secs / dt)
        for i in range(n):
            f = min(1.0, (i + 1) / (n * 0.6))
            p = cur + (goal - cur) * f
            y = cur_yaw + (yaw - cur_yaw) * f
            q_prev = mp.q_cmd.copy()
            q, wdt = mp.step({"pinch": p, "yaw": y - math.pi / 2, "gap": gap}, dt, True)
            arm.send(q, wdt)
            worst_step = max(worst_step, np.max(np.abs(q - q_prev)) / dt)
            arm.joints()
        cur, cur_yaw = goal, yaw
        tip, mat = kin.fk(mp.q_cmd)
        want = mp.human_to_robot(goal)
        tilt = math.degrees(math.acos(np.clip(-mat[2, 2], -1, 1)))
        check(f"4 집게 끝이 목표 {np.round(goal * 100, 1)} cm 에 2 mm 안", np.linalg.norm(tip - want) < 0.002)
        close_dir = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))          # 집게가 닫히는 줄의 방향 (사이트 x 축)
        print(f"  사람 {np.round(goal * 100, 1)} cm, 엄지↔손가락 {gap * 1000:.0f} mm, 손 줄 {math.degrees(yaw):+.0f}° → "
              f"집게 끝 {np.round((tip - [0, 0, TABLE_Z]) * 100, 1)} cm (목표와 {np.linalg.norm(tip - want) * 1000:.1f} mm), "
              f"닫히는 줄 {axis_err_deg(math.radians(close_dir), yaw):+.0f}° 차, 기울기 {tilt:.0f}°, 집게 폭 {wdt * 1000:.0f} mm")
    print(f"  관절 최고 속도 {worst_step:.2f} rad/s (제한 {mp.max_joint_rate})")
    check("4 관절 속도 제한 지킴", worst_step <= mp.max_joint_rate + 1e-6)

    print("== 5. 손 돌림 훑기: 손 줄을 오른쪽 45° → 왼쪽 110° 로 천천히 (손 벌림 9 cm, 높이 5 cm)")
    for name, p in (("종이 가운데", (0.0, 0.0, 0.05)), ("왼쪽 앞", (0.08, 0.12, 0.05)), ("오른쪽 뒤", (-0.08, -0.12, 0.05))):
        mp = Mapper(kin)
        mp.sync_to(Preview().joints())
        p = np.array(p)
        for _ in range(90):                                                    # 자리 잡기
            mp.step({"pinch": p, "yaw": math.radians(-45) - math.pi / 2, "gap": 0.09}, dt, True)
        errs, flips, prev = [], 0, mp.yaw_goal
        for deg in np.arange(-45, 110.01, 0.5):                                # 초당 15°
            for _ in range(1):
                mp.step({"pinch": p, "yaw": math.radians(deg) - math.pi / 2, "gap": 0.09}, dt, True)
            flips += abs(mp.yaw_goal - prev) > math.radians(60)
            prev = mp.yaw_goal
            _, mat = kin.fk(mp.q_cmd)
            errs.append(abs(axis_err_deg(math.atan2(mat[1, 0], mat[0, 0]), math.radians(deg))))
        errs = np.array(errs)
        check(f"5 {name}: 집게가 반 바퀴 돌지 않음", flips == 0)
        print(f"  {name:8s} 닫히는 줄 차이: 중간값 {np.median(errs):.0f}°, 10° 넘는 때 {np.mean(errs > 10) * 100:.0f}%, "
              f"최대 {errs.max():.0f}°, 반 바퀴 돎 {flips}번")


def check_hard_cases():
    """11/16 검토에서 더한 시험: 엉뚱한 첫 장면, MediaPipe 손 3D 모양 오차, 손 3D 가 거울상일 때."""
    from geometry import flat_fit
    from teleop import Smoother
    print("== 6. 손을 다시 찾은 첫 장면이 엉뚱할 때 (x 40 cm 한 장 → 참 x 0)")
    sm = Smoother()
    seq = [{"pinch": np.array([0.40, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] + \
          [{"pinch": np.array([0.0, 0.0, 0.05]) + RNG.normal(0, 0.003, 3), "yaw": 0.0, "gap": 0.08} for _ in range(30)]
    outs = [sm(i / 30, m) for i, m in enumerate(seq)]
    first = next(i for i, o in enumerate(outs) if o is not None)
    print(f"  처음 내보낸 장면 {first} (x {outs[first]['pinch'][0] * 100:.1f} cm), 1초 뒤 x {outs[-1]['pinch'][0] * 100:.1f} cm, "
          f"엉뚱한 값을 내보낸 장면 {sum(o is not None and o['pinch'][0] > 0.2 for o in outs)}개")
    check("6 엉뚱한 첫 장면을 내보내지 않음", sum(o is not None and o['pinch'][0] > 0.2 for o in outs) == 0)
    sm = Smoother()
    seq = [{"pinch": np.array([0.0, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] * 10 + \
          [{"pinch": np.array([0.20, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08}] * 10
    outs = [sm(i / 30, m) for i, m in enumerate(seq)]
    print(f"  손이 정말 20 cm 옮겨 가면: 옮긴 뒤 {next(i for i, o in enumerate(outs[10:]) if o and o['pinch'][0] > 0.1) + 1}장째에 따라감")

    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    R, t = look_at(np.array([0.65, 0.0, 0.25]), np.array([-0.05, 0.0, 0.05]))
    calib = TableCalib(shoot(R, t, K_true, CORNERS_TABLE, 1.0), W, H)

    def obs(kind, pos, yaw, w, wn, mirror=False):
        local = hand_model(kind, w)
        img = shoot(R, t, K_true, (rotz(yaw) @ local.T).T + np.asarray(pos, float), 2.0)
        world = local * 1.10 + RNG.normal(0, wn, local.shape)
        world -= world.mean(axis=0)
        if mirror:
            world[:, 2] *= -1
        return world, img

    print("== 7. MediaPipe 손 3D 모양이 점마다 틀릴 때 (화면 점 2 px 흔들림, 다섯 자리 × 40번)")
    for wn in (0.0, 0.003, 0.006, 0.010):
        s = scale_for_flat(calib, np.median([HandMeasure(calib, 1.0)(*obs("flat", (0, 0, 0), 0, 0, wn))["lm_cam_raw"]
                                             for _ in range(20)], axis=0))
        meas = HandMeasure(calib, s)
        errs, yerr = [], []
        for p, yaw, w in (((0, 0, 0.03), 0.0, 0.08), ((0.15, 0.10, 0.03), 0.3, 0.055), ((0.05, -0.05, 0.12), 0.4, 0.055),
                          ((-0.08, 0.12, 0.05), -0.4, 0.07), ((0.10, -0.12, 0.15), 0.0, 0.10)):
            for _ in range(40):
                r = meas(*obs("grasp", p, yaw, w, wn))
                errs.append(np.linalg.norm(r["pinch"] - p))
                yerr.append(abs(axis_err_deg(r["yaw"], yaw - math.pi / 2)))
        if wn == 0.0:
            check("7 손 3D 모양이 맞을 때 집기점 중간값 0.6 cm 안", np.median(errs) < 0.006)
        print(f"  점마다 {wn * 1000:2.0f} mm 틀림: 배율 {s:.3f} (참 0.909), 집기점 오차 중간값 {np.median(errs) * 100:.1f} cm "
              f"/ 열에 아홉은 {np.percentile(errs, 90) * 100:.1f} cm 안, 집게 방향 {np.median(yerr):.0f}° / {np.percentile(yerr, 90):.0f}°")

    print("== 8. 손 3D 가 거울상일 때 보정에서 알아채는가 (20장 다수결, 30번)")
    for wn in (0.003, 0.006):
        for truth in (False, True):
            right = 0
            for _ in range(30):
                votes = []
                for _ in range(20):
                    w, img = obs("flat", (0, 0, 0), RNG.uniform(-0.4, 0.4), 0, wn, truth)
                    r0, r1 = HandMeasure(calib, 1.0, mirror=False)(w, img), HandMeasure(calib, 1.0, mirror=True)(w, img)
                    votes.append(flat_fit(calib, r1["lm_cam_raw"])[1] < flat_fit(calib, r0["lm_cam_raw"])[1])
                right += (sum(votes) > 0.7 * len(votes)) == truth
            print(f"  점마다 {wn * 1000:.0f} mm 틀림, 실제로 거울상 {str(truth):5s}: 30번 중 {right}번 맞게 판단")
            check(f"8 거울상 판별 ({wn * 1000:.0f} mm, {truth}) 30번 중 27번 넘게", right >= 27)


def rot_axis(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def grasp_like_flat(curl=(0.9, 1.1, 0.6), less=0.0, cup=0.0):
    """종이에 편 손(hand_model flat)과 뼈 길이·손바닥이 같은 쥔 손 (손바닥 아래로, 집기점이 원점).
    네 손가락을 뿌리·가운데·끝 마디에서 아래로 굽히고 엄지는 손바닥 아래로 내린다.
    less: 손가락이 덜 굽은 만큼 (rad) — 쥘 때 MediaPipe 가 손가락 모양을 틀리게 내는 것을 흉내. cup: 약지·새끼 뿌리가 오므라듦 (m)."""
    p = hand_model("flat").copy()
    for ch in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        for j in range(3):
            R = rot_axis([0, 1.0, 0], max(curl[j] - less, 0.0))
            base = p[ch[j]].copy()
            for c in ch[j + 1:]:
                p[c] = base + R @ (p[c] - base)
    for c, d in zip((2, 3, 4), ((0.012, -0.006, -0.032), (0.014, -0.008, -0.026), (0.012, -0.008, -0.022))):
        p[c] = p[c - 1] + np.array(d) * (1 + 0.6 * less)
    if cup:
        p[13] += np.array([0, cup, -cup * 0.5]) * 0.5
        p[17] += np.array([0, cup, -cup * 0.5])
    return p - 0.5 * (p[4] + p[[8, 12, 16, 20]].mean(axis=0))


def check_finger_shape():
    """11/18: 11/16 실물에서 손 높이가 책상 아래 5~20 cm 로 나온 까닭을 흉내. 쥘 때 MediaPipe 손 3D 의 손가락이 덜 굽게 나오면
    21점으로 방향·위치를 다 푸는 pnp 는 그 틀림이 카메라 쪽 거리로 커지고, 방향은 MediaPipe 것을 쓰고 위치만 푸는 oat 는 덜 커진다.
    MediaPipe 손 3D 는 카메라와 같은 방향으로 준다 (OAT 의 가정, 11/17 예시 사진에서 1~5° 안). 여기서는 5° 를 일부러 틀리게 준다."""
    from geometry import backproject_height, palm_size
    print("== 9. 쥘 때 손가락 모양이 틀릴 때 (손 방향 5° 틀림, 점마다 3 mm, 화면 2 px, 다섯 자리 × 12번)")
    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    R, t = look_at(np.array([0.65, 0.0, 0.25]), np.array([-0.05, 0.0, 0.05]))
    calib = TableCalib(shoot(R, t, K_true, CORNERS_TABLE, 1.0), W, H)
    rng = np.random.default_rng(3)

    def flat_obs():
        local = hand_model("flat")
        world = (R @ (local * 1.10).T).T + rng.normal(0, 0.003, local.shape)
        return world - world.mean(axis=0), shoot(R, t, K_true, local, 2.0)

    def grasp_obs(pos, yaw, tilt, less, cup):
        Rh = rotz(yaw) @ rot_axis([0, 1.0, 0], tilt)
        img = shoot(R, t, K_true, (Rh @ grasp_like_flat(cup=cup).T).T + np.asarray(pos, float), 2.0)
        Re = rot_axis(rng.normal(0, 1, 3), math.radians(5))
        world = (Re @ R @ Rh @ (grasp_like_flat(less=less) * 1.10).T).T + rng.normal(0, 0.003, (21, 3))
        return world - world.mean(axis=0), img

    flats = [flat_obs() for _ in range(20)]
    flat21 = np.median([backproject_height(calib, o[1], FLAT_H) for o in flats], axis=0)
    (pw, pl), (tw, tl) = palm_size(flat21), palm_size(hand_model("flat"))
    print(f"  손바닥 보정: 폭 {pw * 100:.2f} cm (참 {tw * 100:.2f}), 길이 {pl * 100:.2f} cm (참 {tl * 100:.2f})")
    check("9 종이에 편 손으로 잰 손바닥 크기 2 mm 안", abs(pw - tw) < 0.002 and abs(pl - tl) < 0.002)
    ms = {}
    for solver in ("pnp", "oat"):
        cams = [HandMeasure(calib, 1.0, solver=solver)(*o)["lm_cam_raw"] for o in flats]
        ms[solver] = HandMeasure(calib, scale_for_flat(calib, np.median(cams, axis=0)), solver=solver)
    ms["palm"] = HandMeasure(calib, 1.0, solver="palm", flat=flat21)
    spots = [((0, 0, 0.03), 0.0, 0.0), ((0.15, 0.10, 0.03), 0.3, 0.2), ((0.05, -0.05, 0.12), 0.4, -0.2),
             ((0.10, -0.12, 0.20), 0.0, 0.3), ((-0.08, 0.12, 0.05), -0.4, 0.0)]
    res = {}
    for cup in (0.0, 0.01):
        for less_deg in (0, 15, 30):
            row = []
            for name, m in ms.items():
                e, ez = [], []
                for pos, yaw, tilt in spots:
                    for _ in range(12):
                        r = m(*grasp_obs(pos, yaw, tilt, math.radians(less_deg), cup))
                        if r is not None:
                            e.append(np.linalg.norm(r["pinch"] - pos))
                            ez.append(r["pinch"][2] - pos[2])
                res[(cup, less_deg, name)] = (np.median(e), np.median(np.abs(ez)))
                row.append(f"{name} {np.median(e) * 100:4.1f} cm (높이 {np.median(ez) * 100:+4.1f})")
            print(f"  손바닥 오므림 {cup * 100:.0f} cm, 손가락 {less_deg:2d}° 덜 굽게: 집기점 오차 중간값  " + " | ".join(row))
    check("9 손가락 모양이 15·30° 틀릴 때 oat 집기점 오차가 pnp 의 절반 아래",
          all(res[(c, l, "oat")][0] < 0.5 * res[(c, l, "pnp")][0] for c in (0.0, 0.01) for l in (15, 30)))
    check("9 oat: 손가락 30° 틀려도 높이 오차 중간값 2.5 cm 안",
          all(res[(c, 30, "oat")][1] < 0.025 for c in (0.0, 0.01)))


def check_auto_calib():
    """11/18 저절로 보정: 그린 책상 사진에서 A4 모서리 찾기, 손을 + 위에 펴서 가만히 두었는지 알아채기."""
    import cv2
    from fake_scene import project, render_desk
    from geometry import FlatHandGate
    from paper import find_paper
    print("== 10. 종이 저절로 찾기 (그린 책상 사진 1280x720, 카메라 자리 6 × 책상 색 3)")
    K_true = intrinsics(focal_from_hfov(70.0, W), W, H)
    rng = np.random.default_rng(5)
    poses = {"노트북 정면 (앞 65, 높이 25)": ((0.65, 0, 0.25), (-0.05, 0, 0.05)),
             "노트북 낮게 (앞 55, 높이 18)": ((0.55, 0.0, 0.18), (-0.05, 0, 0.03)),
             "노트북 조금 비낌": ((0.60, -0.05, 0.28), (0.0, 0.0, 0.03)),
             "C920 위 (앞 45, 높이 55)": ((0.45, 0.0, 0.55), (0.0, 0, 0.0)),
             "비스듬히 (앞 50 왼 30)": ((0.50, 0.30, 0.40), (0.0, 0, 0.0)),
             "옆으로 돌림 (앞 60 오른 20)": ((0.60, -0.20, 0.28), (0.0, 0.0, 0.03))}
    worst, wrong, missed, cam_err = 0.0, 0, 0, 0.0
    for name, (cp, tgt) in poses.items():
        R, t = look_at(np.array(cp), np.array(tgt))
        true = project(R, t, K_true, CORNERS_TABLE)
        errs = []
        for desk in ((60, 85, 115), (150, 150, 150), (205, 205, 200)):
            r = find_paper(render_desk(R, t, K_true, W, H, rng, desk=desk))
            if r is None:
                missed += 1
                continue
            e = float(np.max(np.linalg.norm(r[0] - true, axis=1)))
            wrong += e > 20                                     # 순서가 틀리면 수십~수백 px
            worst = max(worst, e)
            cam_err = max(cam_err, float(np.linalg.norm(r[1].cam_pos - np.array(cp))))
            errs.append(e)
        print(f"  {name:22s} 찾음 {len(errs)}/3, 모서리 오차 최대 {max(errs) if errs else float('nan'):.1f} px")
    print(f"  모두: 못 찾음 {missed}, 순서 틀림 {wrong}, 모서리 오차 최대 {worst:.1f} px, 카메라 자리 오차 최대 {cam_err * 100:.1f} cm")
    check("10 종이 저절로 찾기: 18장 모두 찾고 순서 맞음, 모서리 3 px 안", missed == 0 and wrong == 0 and worst < 3.0)
    check("10 종이 저절로 찾기: 카메라 자리 3 cm 안", cam_err < 0.03)
    R, t = look_at(np.array([0.65, 0.0, 0.25]), np.array([-0.05, 0.0, 0.05]))
    empty = render_desk(R, t, K_true, W, H, rng)
    empty[:] = cv2.GaussianBlur(np.full_like(empty, 90), (3, 3), 0)
    sq = render_desk(R, t, K_true, W, H, rng, other=[(-0.30, 0.25), (-0.30, 0.10), (-0.15, 0.10), (-0.15, 0.25)])
    sq_only = render_desk(R, t, K_true, W, H, rng, other=[(-0.05, 0.08), (-0.05, -0.07), (0.10, -0.07), (0.10, 0.08)])
    covered = render_desk(R, t, K_true, W, H, rng, blob=(A4_D / 2, -A4_W / 2, 0.05))
    r_sq = find_paper(sq)
    print(f"  종이 없음: {'못 찾음' if find_paper(empty) is None else '찾았다고 함'} / 종이 + 옆에 흰 정사각 카드: "
          f"{'A4 를 찾음' if r_sq is not None and np.max(np.linalg.norm(r_sq[0] - project(R, t, K_true, CORNERS_TABLE), axis=1)) < 3 else '틀림'}"
          f" / 모서리 하나를 손이 가림: {'못 찾음 (손을 치우면 찾음)' if find_paper(covered) is None else '찾음'}")
    calib = TableCalib(project(R, t, K_true, CORNERS_TABLE), W, H)
    r_only = find_paper(_paint_square_only(sq_only, R, t, K_true))
    check("10 종이 없거나 정사각 카드만 있으면 찾았다고 하지 않음", find_paper(empty) is None and r_only is None)
    r_cov = find_paper(covered)
    check("10 손이 모서리를 가리면 엉뚱한 모서리를 받지 않음",
          r_cov is None or np.max(np.linalg.norm(r_cov[0] - project(R, t, K_true, CORNERS_TABLE), axis=1)) < 3)
    check("10 A4 옆에 다른 흰 카드가 있어도 A4 를 고름", r_sq is not None and
          np.max(np.linalg.norm(r_sq[0] - project(R, t, K_true, CORNERS_TABLE), axis=1)) < 3)

    print("== 11. 손을 + 위에 펴서 가만히 두었는지 (가만히 둔 손도 화면 점이 1.5 px 씩 흔들림)")
    gate_rng = np.random.default_rng(6)

    def flat_frames(pos, yaw, n, kind="flat", noise=1.5, drift=0.0):
        local = hand_model(kind, 0.08)
        outs = []
        for i in range(n):
            p = np.asarray(pos, float) + np.array([drift * i, 0, 0])
            img = shoot(R, t, K_true, (rotz(yaw) @ local.T).T + p, 0.0) + gate_rng.normal(0, noise, (21, 2))
            world = (R @ rotz(yaw) @ (local * 1.10).T).T + gate_rng.normal(0, 0.003, (21, 3))
            outs.append((world - world.mean(axis=0), img))
        return outs

    def first_ok(frames):
        g = FlatHandGate(calib, "oat")
        for i, (w_, im) in enumerate(frames):
            ok, why = g(w_, im)
            if ok:
                return i + 1, why
        return None, why

    cases = [("+ 위에 편 손, 가만히", flat_frames((0, 0, 0), 0.0, 30), True),
             ("편 손을 20° 돌려 놓음", flat_frames((0, 0, 0), math.radians(20), 30), True),
             ("+ 에서 8 cm 옆", flat_frames((0, 0.08, 0), 0.0, 30), False),
             ("손가락을 옆으로 (70°)", flat_frames((0, 0, 0), math.radians(70), 30), False),
             ("손을 6 cm 들고 있음", flat_frames((0, 0, 0.06), 0.0, 30), False),
             ("쥔 손", flat_frames((0, 0, 0.03), 0.0, 30, kind="grasp"), False),
             ("+ 근처에서 천천히 미는 중 (장면마다 1.5 mm)", flat_frames((-0.02, 0, 0), 0.0, 26, drift=0.0015), False)]
    good = True
    for name, frames, want in cases:
        n, why = first_ok(frames)
        print(f"  {name:24s}: {'시작함 (' + str(n) + '장째)' if n else '안 함 — ' + why}")
        good &= (n is not None) == want
    check("11 편 손·가만히일 때만 손 보정을 저절로 시작", good)


def _paint_square_only(img, R, t, K):
    """sq_only 사진에서 A4 를 책상색으로 덮어 흰 정사각 카드만 남긴다."""
    import cv2
    from fake_scene import project
    out = img.copy()
    p = project(R, t, K, CORNERS_TABLE * 1.03)
    cv2.fillPoly(out, [np.round(p).astype(np.int32)], (60, 85, 115))
    q = project(R, t, K, [[-0.05, 0.08, 0], [-0.05, -0.07, 0], [0.10, -0.07, 0], [0.10, 0.08, 0]])
    cv2.fillPoly(out, [np.round(q).astype(np.int32)], (215, 218, 220))
    return out


def verdict():
    bad = [n for n, ok in CHECKS if not ok]
    print()
    if bad:
        print(f"결과: 이상 있음 ({len(bad)}/{len(CHECKS)}개 실패)")
        for n in bad:
            print("  실패:", n)
    else:
        print(f"결과: 모두 통과 ({len(CHECKS)}개)")
    return not bad


def check_steady():
    """11/20 지터·손떨림 거르기: 멈춘 손의 떨림과 한 장면 튐을 로봇 목표에 넘기지 않는지, 움직이면 따라가는지,
    로봇 PC 가 패킷 사이를 이어 끊어 가지 않는지."""
    from teleop import Smoother
    print("== 12. 멈춘 손 떨림 (손떨림 9 Hz 1 mm + 잡음 xy 2·z 5 mm, 23 fps, 4초) 과 한 장면 3 cm 튐")
    rng = np.random.default_rng(12)

    def run(sm, spike_at=()):
        outs = []
        for i in range(92):
            t = i / 23
            p = np.array([0.0, 0.0, 0.05]) + 0.001 * np.sin(2 * np.pi * 9 * t + np.arange(3)) + rng.normal(0, [0.002, 0.002, 0.005])
            if i in spike_at:
                p = p + np.array([0.0, 0.03, 0.0])
            o = sm(t, {"pinch": p, "yaw": 0.0, "gap": 0.08})
            outs.append(None if o is None else o["pinch"])
        o = np.array([x for x in outs[23:] if x is not None])
        return float(np.max(np.linalg.norm(o - o.mean(0), axis=1)) * 2000)    # 가장 크게 흔들린 폭 (mm)
    old, new = run(Smoother(steady=False), (50,)), run(Smoother(), (50,))
    print(f"  흔들린 폭 (튐 포함): 1€ 만 {old:.1f} mm → 새 거르기 {new:.1f} mm")
    check("12 멈춘 손 떨림·튐을 8 mm 안으로 (1€ 만일 때의 0.4 배 아래)", new < 8.0 and new < old * 0.4)
    # 움직일 때: 0.3 m/s 로 20 cm. 끝나고 0.5초 안에 1 cm 안으로 따라오고 넘치지 않는지
    sm, outs = Smoother(), []
    for i in range(60):
        t = i / 23
        x = min(0.3 * max(t - 0.5, 0.0), 0.20)
        o = sm(t, {"pinch": np.array([x, 0.0, 0.05]), "yaw": 0.0, "gap": 0.08})
        outs.append(0.0 if o is None else o["pinch"][0])
    t_end = 0.5 + 0.20 / 0.3
    after = [o for i, o in enumerate(outs) if i / 23 > t_end + 0.5]
    lag = 0.3 * max(t_end - 0.1 - 0.5, 0) - outs[int((t_end - 0.1) * 23)]
    print(f"  0.3 m/s 이동 중 뒤처짐 {lag * 1000:.1f} mm, 멈추고 0.5초 뒤 오차 {abs(after[0] - 0.20) * 1000:.1f} mm, "
          f"넘침 {max(0.0, max(outs) - 0.20) * 1000:.1f} mm")
    check("12 움직이면 따라가고 멈추면 1 cm 안", abs(after[0] - 0.20) < 0.01 and max(outs) < 0.205 and lag < 0.03)

    print("== 13. 로봇 PC 패킷 사이 잇기 (손 0.2 m/s 등속, 패킷 23 Hz, 팔 100 Hz)")
    kin = PiperKin()

    def robot(interp):
        mp = Mapper(kin, max_speed=1.0, max_joint_rate=3.0)
        mp.interp = interp
        mp.sync_to(np.array([0.0, 0.9, -0.9, 0.0, 0.9, 0.0]))
        pkt, curs = None, []
        for k in range(150):
            t = k / 100
            n = int(t * 23)
            x = -0.03 + 0.2 * min(n / 23, 0.6)
            if pkt is None or pkt["n"] != n:
                pkt = {"n": n, "pinch": np.array([x, 0.0, PINCH_H0]), "yaw": 0.0, "gap": 0.0}   # 새 패킷 = 새 dict
            mp.step(pkt, 0.01, True, ik_iters=10)
            curs.append(mp.cur.copy())
        c = np.array(curs)
        v = np.linalg.norm(np.diff(c, axis=0), axis=1) * 100
        mid = v[30:55]                                   # 등속 구간
        return float(mid.max() / max(mid.mean(), 1e-9)), float(np.linalg.norm(c[-1] - mp.human_to_robot(pkt["pinch"])) * 1000)
    (r_old, e_old), (r_new, e_new) = robot(False), robot(True)
    print(f"  등속 구간 최고 속도 / 평균 속도: 잇기 없음 {r_old:.1f} → 잇기 {r_new:.1f} (1 이면 고르게), 마지막 목표와 차 {e_new:.1f} mm")
    check("13 패킷 사이를 고르게 잇고 마지막 값에 닿음", r_new < 1.6 and r_new < r_old * 0.6 and e_new < 1.0)


def check_palm_tilt():
    """11/23 손바닥 방향 → 집게 방향. 손등이 보이면(손바닥 아래) 집게 아래, 손바닥이 위면 집게도 위로 (관절이 닿는 데까지)."""
    print("== 14. 손바닥 방향 따라 집게 돌리기 (가상 팔)")
    import geometry
    from mapping import Mapper, approach_from_palm, min_height_for
    from piper_kin import PiperKin, TABLE_Z, grip_orientation
    kin = PiperKin()

    def rot_x(a):                                   # 손을 앞 축(x)으로 돌림 = 손목을 뒤집음
        c, s_ = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])

    flat = hand_model("flat")
    n_down = geometry.palm_normal(flat)
    n_up = geometry.palm_normal((rot_x(math.pi) @ flat.T).T)
    print(f"  펴서 엎은 오른손 손바닥 쪽 {np.round(n_down, 2)}, 뒤집은 손 {np.round(n_up, 2)}")
    check("14a 손등이 보이면 손바닥 쪽이 아래, 뒤집으면 위", n_down[2] < -0.95 and n_up[2] > 0.95)
    check("14b 손바닥이 20° 안으로 기울면 집게는 똑바로 아래", approach_from_palm(rot_x(math.radians(15)) @ np.array([0, 0, -1.0]))[2] < -0.999)

    def run(tilt_deg, h, steps=400, down_only=False):
        mp = Mapper(kin, max_speed=0.3, max_joint_rate=3.0, max_yaw_rate=math.radians(120))
        mp.follow_tilt = not down_only
        mp.sync_to(np.array([0.0, 0.9, -0.9, 0.0, 0.9, 0.0]))
        R = rot_x(math.radians(tilt_deg))
        meas = {"pinch": np.array([0.0, 0.0, PINCH_H0 + h]), "yaw": 0.0, "gap": 0.08,
                "normal": R @ np.array([0, 0, -1.0]), "axis": R @ np.array([0, 1.0, 0])}
        low = 9.0
        for _ in range(steps):
            q, _w = mp.step(meas, 0.01, True, ik_iters=10)
            pts = np.asarray(kin.link_points(q))[1:-1]
            low = min(low, float(pts[:, 2].min()) - TABLE_Z)
        p, m = kin.fk(mp.q_cmd)
        tilt = math.degrees(math.acos(float(np.clip(-m[2, 2], -1, 1))))     # 0 아래, 180 위
        jaw_ok = p[2] >= min_height_for(m, TABLE_Z, 0.0) - 0.01
        return tilt, (p[2] - TABLE_Z) * 100, low * 100, jaw_ok, mp

    t0, z0, l0, j0, mp0 = run(0, 0.05)
    _, m_old = kin.fk(mp0.q_cmd)
    want = grip_orientation(0.0)
    err = min(math.degrees(math.acos(float(np.clip((np.trace(m_old @ (want @ F).T) - 1) / 2, -1, 1))))
              for F in (np.eye(3), np.diag([-1.0, -1.0, 1.0])))
    print(f"  손바닥 아래: 집게 {t0:.0f}° (예전 방식과 차 {err:.1f}°), 집게 끝 높이 {z0:.1f} cm")
    check("14c 손바닥이 아래면 예전처럼 똑바로 아래", t0 < 3 and err < 3)
    rows = []
    for tilt_h, h in ((90, 0.05), (90, 0.15), (180, 0.05), (180, 0.20)):
        t_, z_, l_, j_, mp_ = run(tilt_h, h)
        rows.append((tilt_h, h, t_, z_, l_, j_, mp_.blocked))
        print(f"  손바닥 {tilt_h:3d}° (0 아래, 180 위), 손 높이 +{h*100:.0f} cm → 집게 {t_:3.0f}°, 집게 끝 {z_:4.1f} cm, "
              f"가장 낮은 팔 마디 {l_:+5.1f} cm, 집게 몸통·손가락 책상 위 {'예' if j_ else '아니오'}{' (멈춤 있었음)' if mp_.blocked else ''}")
    # PiPER 는 손목 꺾기 ±70° 라 책상 가까이에서 집게가 옆으로 세워지는 데 한계가 있다 (11/23 관절 6만 개 뽑아 봄: 높이 16 cm 아래 70~120°)
    check("14d 손바닥을 옆으로 세우면 집게도 옆으로 (45° 넘게)", rows[0][2] > 45 and rows[1][2] > 45)
    check("14e 손바닥을 위로 뒤집으면 옆으로 세울 때보다 더 위로", max(r[2] for r in rows[2:]) > max(rows[0][2], rows[1][2]))
    check("14f 어느 쪽으로 돌려도 팔 마디와 집게가 책상에 닿지 않음", all(r[4] > 0.5 and r[5] for r in rows))
    td, _, _, _, _ = run(180, 0.05, down_only=True)
    check("14g --down-only 면 손바닥을 뒤집어도 집게는 아래", td < 3)
    # 돌리는 도중 반 바퀴 휙 돌지 않음: 손을 0 → 180° 천천히 뒤집는 동안 한 박자 집게 회전이 작다
    mp = Mapper(kin, max_speed=0.3, max_joint_rate=3.0, max_yaw_rate=math.radians(120))
    mp.sync_to(np.array([0.0, 0.9, -0.9, 0.0, 0.9, 0.0]))
    prev, worst = None, 0.0
    for k in range(600):
        R = rot_x(math.radians(180 * min(k / 400, 1.0)))
        meas = {"pinch": np.array([0.0, 0.0, PINCH_H0 + 0.15]), "yaw": 0.0, "gap": 0.08,
                "normal": R @ np.array([0, 0, -1.0]), "axis": R @ np.array([0, 1.0, 0])}
        q, _ = mp.step(meas, 0.01, True, ik_iters=10)
        _, m = kin.fk(q)
        if prev is not None:
            worst = max(worst, math.degrees(math.acos(float(np.clip((np.trace(m @ prev.T) - 1) / 2, -1, 1)))))
        prev = m
    print(f"  손을 4초에 걸쳐 뒤집는 동안 한 박자(10 ms) 집게 회전 최대 {worst:.1f}°")
    check("14h 뒤집는 동안 집게가 휙 돌지 않음 (한 박자 3° 안)", worst < 3.0)


def check_hold_start():
    """11/23 오후 사용자: "스페이스바 누르고 시작하면 집게가 뜨는데? 자세가 그대로여야지". hold_start 면 SPACE 순간 팔이 그대로."""
    print("== 15. SPACE 때 자세 그대로 (hold_start, 가상 팔)")
    from mapping import Mapper, rot_angle
    from piper_kin import PiperKin, TABLE_Z
    kin = PiperKin()
    qh = np.radians([-2.82, 92.996, -26.67, 4.49, 43.984, -7.491])       # 11/23 사용자가 고친 처음 자세 (모델로는 끝이 책상 −0.8 cm, 16° 기울어짐)
    hp, hm = kin.fk(qh)

    def rot_x(a):
        c, s_ = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])

    def rot_z(a):
        c, s_ = math.cos(a), math.sin(a)
        return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])

    def mk(hold):
        mp = Mapper(kin, float(hp[0]), float(hp[1]), 1.0, max_speed=1.0, min_z=0.006, max_joint_rate=3.0,
                    max_yaw_rate=math.radians(180), max_z=0.30, box_x=(0.12, 0.50), box_y=0.40)
        mp.robot_start[2] = max(float(hp[2]), TABLE_Z + 0.006)
        mp.hold_start, mp.interp = hold, True
        mp.sync_to(qh.copy())
        mp.width = 0.0
        return mp

    def meas(pinch, R, gap):
        return {"pinch": np.asarray(pinch, float), "yaw": 0.0, "gap": gap,
                "normal": R @ np.array([0, 0, -1.0]), "axis": R @ np.array([0, 1.0, 0])}

    def go(mp, f, n):
        for k in range(n):
            mp.step(f(k), 0.01, True, ik_iters=10)
        p, m = kin.fk(mp.q_cmd)
        return p, m

    P0, R0, G0 = np.array([0.0, 0.03, PINCH_H0 + 0.05]), rot_z(math.radians(30)) @ rot_x(math.radians(15)), 0.08
    worst = (0.0, 0.0, 0.0)
    for P, R in ((np.array([0, 0, PINCH_H0]), np.eye(3)), (P0, R0), (np.array([0.02, -0.02, PINCH_H0 + 0.02]), rot_x(math.radians(35)))):
        mp = mk(True)
        p, m = go(mp, lambda k: meas(P, R, G0), 200)
        worst = max(worst, (float(np.linalg.norm(p - hp)), math.degrees(rot_angle(m, hm)), mp.width))
    print(f"  손이 + 에서 벗어나 있고 손바닥이 기울고 손이 펴져 있어도: 2초 뒤 집게 끝 {worst[0]*1000:.1f} mm, 방향 {worst[1]:.1f}°, 그리퍼 {worst[2]*100:.1f} cm")
    check("15a SPACE 뒤 손이 어디 있든 팔은 그대로 (1 mm·0.5°·그리퍼 그대로)", worst[0] < 0.001 and worst[1] < 0.5 and worst[2] == 0.0)
    mp = mk(False)
    p, m = go(mp, lambda k: meas([0, 0, PINCH_H0], np.eye(3), G0), 200)
    print(f"  예전(--start paper): 손 + 위 편 손에서 집게 끝 {np.linalg.norm(p - hp)*1000:.0f} mm 뜸, 방향 {math.degrees(rot_angle(m, hm)):.0f}° 돎, 그리퍼 {mp.width*100:.1f} cm")
    check("15b --start paper 는 예전처럼 종이 + 기준 (이 자세에서는 움직임)", np.linalg.norm(p - hp) > 0.005)
    mp = mk(True)
    p, m = go(mp, lambda k: meas(P0 + np.array([0.10, 0.0, 0.05]) * min(k / 150, 1.0), R0, G0), 400)
    print(f"  SPACE 뒤 손을 앞 10·위 5 cm: 집게 끝 {np.round((p - hp) * 100, 1)} cm")
    check("15c 그 뒤로는 손이 움직인 만큼 (5 mm 안)", np.linalg.norm((p - hp) - [0.10, 0.0, 0.05]) < 0.005)
    mp = mk(True)
    p, m = go(mp, lambda k: meas(P0 + np.array([0, 0, -0.03]) * min(k / 100, 1.0), R0, G0), 300)
    check("15d 손을 처음보다 낮게 내려도 집게 끝은 처음 높이 아래로 안 감", p[2] > hp[2] - 0.002)
    mp = mk(True)
    go(mp, lambda k: meas(P0, R0, G0 - 0.005), 100)
    w_small = mp.width
    go(mp, lambda k: meas(P0, R0, G0 - 0.04), 100)
    print(f"  손가락 0.5 cm 움직임 → 그리퍼 {w_small*100:.1f} cm, 4 cm 오므림 → {mp.width*100:.1f} cm")
    check("15e 그리퍼는 손가락이 1 cm 넘게 움직이면 따라감", w_small == 0.0 and mp.width > 0.01)
    mp = mk(True)
    for _ in range(50):
        mp.step(None, 0.01, True, ik_iters=10)
    p, m = go(mp, lambda k: meas([0.05, -0.04, PINCH_H0 + 0.08], rot_x(math.radians(25)), 0.07), 200)
    check("15f 손을 늦게 찾아도 (0.5초 뒤) 팔은 그대로", np.linalg.norm(p - hp) < 0.001 and math.degrees(rot_angle(m, hm)) < 0.5)


def check_safe_home():
    """11/23 오후 사용자: "B 를 누르면 로봇팔이 책상에 박고 힘이 빠지네". B 로 갈 자세를 책상 위 3 cm (모델) 까지 들어 올림."""
    print("== 16. B 로 갈 자세가 책상에 박히지 않게 (가상 팔)")
    from home import HOME_CLEAR, path_ok, safe_target
    from mapping import rot_angle
    from piper_kin import PiperKin, TABLE_Z
    kin = PiperKin()
    q_saved = np.radians([-2.82, 92.996, -26.67, 4.49, 43.984, -7.491])     # 11/23 13:15 저장 (모델로 끝 −0.8 cm)
    q_crash = np.radians([-1.13, 86.64, -27.94, 1.12, 42.69, -2.13])       # 실물이 책상에 닿아 힘이 빠진 자세 (대기)
    q_t, lift = safe_target(kin, q_saved)
    p0, m0 = kin.fk(q_saved)
    p1, m1 = kin.fk(q_t)
    print(f"  저장한 자세 끝 {(p0[2] - TABLE_Z) * 100:.1f} cm → B 목표 {(p1[2] - TABLE_Z) * 100:.1f} cm (들어 올림 {lift * 100:.1f} cm), "
          f"앞뒤·옆 차 {np.linalg.norm(p1[:2] - p0[:2]) * 1000:.1f} mm, 방향 차 {math.degrees(rot_angle(m0, m1)):.2f}°")
    check("16a 집게 끝이 낮게 저장된 자세는 같은 방향으로 책상 위 3 cm 까지 들어 올림",
          abs(p1[2] - TABLE_Z - HOME_CLEAR) < 0.001 and np.linalg.norm(p1[:2] - p0[:2]) < 0.002 and rot_angle(m0, m1) < math.radians(1))
    ok_old, low_old = path_ok(kin, q_crash, q_saved)
    ok_new, low_new = path_ok(kin, q_crash, q_t)
    print(f"  힘이 빠진 자세에서 B: 저장한 자세로는 {low_old * 100:.1f} cm 까지 내려감 ({'감' if ok_old else '안 감'}), 들어 올린 자세로는 {low_new * 100:.1f} cm ({'감' if ok_new else '안 감'})")
    check("16b 책상 아래로 가는 길은 안 가고, 들어 올린 자세로는 감", (not ok_old) and ok_new)
    q_hi = np.radians([0.0, 60.0, -60.0, 0.0, 60.0, 0.0])
    q_same, lift0 = safe_target(kin, q_hi)
    check("16c 이미 충분히 높게 저장한 자세는 그대로", lift0 == 0.0 and np.allclose(q_same, q_hi))


def check_follower():
    """11/26 석사형 메모: follower 가 줄에서 꺼내 200 Hz 로 보냄, 3차 중간 점, MIT 모드."""
    print("== 17. follower (줄에서 꺼내 보내기), 3차 중간 점, MIT (가상 팔)")
    from mapping import cubic_s
    from stream import Follower

    class FakeArm:
        is_real = False

        def __init__(self):
            self.q = np.zeros(6)
            self.calls = []

        def joints(self):
            return self.q.copy()

        def gripper(self):
            return 0.05

        def move_joints(self, q):
            self.calls.append(("pos", np.asarray(q).copy(), None))

        def move_mit(self, q, dq, kp, kd):
            self.calls.append(("mit", np.asarray(q).copy(), np.asarray(dq).copy(), kp, kd))

        def send_gripper(self, w):
            pass

        def close(self):
            pass

    eps = 1e-3
    d0 = (cubic_s(eps) - cubic_s(0.0)) / eps
    d1 = (cubic_s(1.0) - cubic_s(1.0 - eps)) / eps
    check("17a 3차 s(t)=3t²−2t³: s(0)=0, s(1)=1, 처음·끝 속도 거의 0, 가운데가 가장 빠름 (1.5배)",
          cubic_s(0) == 0 and cubic_s(1) == 1 and d0 < 0.01 and d1 < 0.01
          and abs((cubic_s(0.5 + eps) - cubic_s(0.5 - eps)) / (2 * eps) - 1.5) < 1e-3)

    arm = FakeArm()
    f = Follower(arm, hz=200, mode="mit", kp=5.0, kd=0.8, push_hz=100, max_rate=3.0, start=False)
    q1 = np.full(6, 0.01)                         # 10 ms 동안 0.01 rad = 1 rad/s
    f.push(q1)
    f.tick(0.000)
    f.tick(0.005)
    a1, a2 = arm.calls[0], arm.calls[1]
    check("17b 100 Hz 목표 하나를 200 Hz 점 두 개로 나눔 (반, 끝)", np.allclose(a1[1], 0.005) and np.allclose(a2[1], 0.01))
    check("17c MIT 로 보낼 때 목표 속도 = 1 rad/s, kp 5, kd 0.8", np.allclose(a1[2], 1.0) and a1[3] == 5.0 and a1[4] == 0.8)
    f.tick(0.010)
    a3 = arm.calls[2]
    check("17d 다음 목표가 안 오면 마지막 점을 속도 0 으로 (그 자리, 앞질러 가지 않음)",
          np.allclose(a3[1], 0.01) and np.allclose(a3[2], 0.0))

    for k in range(10):                           # 다음 목표들이 한꺼번에 몰려 옴 (UDP 가 몰려 온 것처럼)
        f.push(np.full(6, 0.01 + 0.002 * (k + 1)))
    qlen = len(f.q)
    check("17e 몰려 와도 줄은 20 ms 어치(4점)까지만, 넘친 옛 점은 버림 (늦음이 쌓이지 않음)", qlen == 4 and f.n_drop == 16)
    sent = []
    for k in range(4):
        f.tick(0.015 + 0.005 * k)
        sent.append(arm.calls[-1][1][0])
    steps = np.diff([0.01] + sent)
    check("17f 남은 점은 순서대로, 한 점에 관절 0.015 rad(3 rad/s × 5 ms) 넘게 안 바뀜",
          np.all(steps > 0) and np.all(steps <= 0.015 + 1e-9) and abs(sent[-1] - 0.03) < 1e-9)

    f.push(np.full(6, 0.03 + 0.5))                # 한 번에 0.5 rad 튄 목표 (있어서는 안 되지만)
    f.tick(0.040)
    jump = arm.calls[-1][1][0] - 0.03
    check("17g 튄 목표도 한 박자에 0.015 rad 만 감 (MIT 는 팔 안에서 안 잘라 주므로 여기서 자름)", abs(jump - 0.015) < 1e-9 and f.n_clip >= 1)

    arm.q = np.full(6, 0.2)
    f.hold()
    f.tick(0.045)
    check("17h hold(): 줄을 비우고 지금 실제 자세 쪽으로 속도 0 (한 박자 0.015 rad 씩)",
          len(f.q) == 0 and np.allclose(arm.calls[-1][2], 0.0, atol=3.01) and arm.calls[-1][1][0] > 0.045)

    arm2 = FakeArm()
    f2 = Follower(arm2, hz=200, mode="position", push_hz=100, start=False)
    f2.push(np.full(6, 0.004))
    f2.tick(0.0)
    check("17i position 모드는 move_joints 로 보냄 (MIT 아님)", arm2.calls[0][0] == "pos")

    arm3 = FakeArm()
    f3 = Follower(arm3, hz=200, mode="mit", push_hz=100)       # 진짜 쓰레드로 0.5초
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        f3.push(np.full(6, 0.001 * (time.perf_counter() - t0)))
        time.sleep(0.01)
    f3.close()
    n = len(arm3.calls)
    print(f"  쓰레드 0.5초: 보낸 명령 {n}개 (200 Hz 면 100), {f3.report()}")
    check("17j 쓰레드가 실제로 200 Hz 가까이 보냄 (0.5초에 80개 넘게)", n > 80)

    mp = Mapper(PiperKin(), 0.30, 0.0, 1.0, max_speed=1.0, max_joint_rate=3.0)
    mp.interp, mp.profile = True, "cubic"
    mp.hold_start = False
    mp.sync_to(mp.q_cmd)
    hand = {"pinch": np.array([0.0, 0.0, PINCH_H0]), "yaw": -1.047, "gap": 0.07}
    for _ in range(300):                          # 출발 자리에 다 올 때까지 (속도 제한에 걸린 구간을 빼고 재려고)
        mp.step(hand, 0.01, True, ik_iters=10)
    start = mp.cur.copy()
    hand2 = {"pinch": np.array([0.02, 0.0, PINCH_H0]), "yaw": -1.047, "gap": 0.07}
    mp._pkt_dt, mp._since = 0.04, 0.04            # 패킷 간격 40 ms 로 맞춰 둠 (새 패킷에서 0.8·0.04 + 0.2·0.04)
    xs = []
    for _ in range(6):
        mp.step(hand2, 0.01, True, ik_iters=10)
        xs.append(mp.cur[0] - start[0])
    dx = np.diff([0.0] + xs)
    print(f"  mapper cubic: 10 ms 마다 앞으로 {', '.join(f'{v * 1000:.1f}' for v in dx)} mm (40 ms 동안 2 cm)")
    check("17k mapper --profile cubic: 손 값 사이를 느리게 시작 → 빨라짐 → 느리게 끝 (s = 3t² − 2t³)",
          dx[0] < dx[1] and dx[1] > dx[3] and abs(xs[3] - 0.02) < 1e-3 and abs(dx[4]) < 1e-4)


if __name__ == "__main__":
    main()
    check_hard_cases()
    check_finger_shape()
    check_auto_calib()
    check_steady()
    check_palm_tilt()
    check_hold_start()
    check_safe_home()
    check_follower()
    raise SystemExit(0 if verdict() else 1)
