"""numpy 운동학(piper_kin.py)이 mujoco 계산과 같은지 확인한다 (2026-11-15). mujoco 가 있는 환경에서만 돈다.

실행: python check_kin.py
"""
import numpy as np
import mujoco

from piper_kin import HERE, JOINTS, LINKS, SITE, PiperKin, grip_orientation


def main():
    m = mujoco.MjModel.from_xml_path(str(HERE / "kin" / "piper_kin.xml"))
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qadr, dadr = [m.jnt_qposadr[j] for j in jid], [m.jnt_dofadr[j] for j in jid]
    links = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINKS]
    kin = PiperKin()
    rng = np.random.default_rng(0)
    e_pos = e_mat = e_jac = e_link = e_lim = 0.0
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(2000):
        q = rng.uniform(kin.lo, kin.hi)
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jp, jr, site)
        Jp, Jo, pos, mat = kin.jacobian(q)
        e_pos = max(e_pos, np.abs(pos - d.site_xpos[site]).max())
        e_mat = max(e_mat, np.abs(mat - d.site_xmat[site].reshape(3, 3)).max())
        e_jac = max(e_jac, np.abs(Jp - jp[:, dadr]).max(), np.abs(Jo - jr[:, dadr]).max())
        e_link = max(e_link, np.abs(kin.link_points(q)[:-1] - d.xpos[links]).max())
    e_lim = max(np.abs(kin.lo - (m.jnt_range[jid][:, 0] + 0.01)).max(), np.abs(kin.hi - (m.jnt_range[jid][:, 1] - 0.01)).max())
    print(f"2000 자세: 집게 끝 위치 차 {e_pos:.1e} m, 방향 차 {e_mat:.1e}, 자코비안 차 {e_jac:.1e}, "
          f"링크 위치 차 {e_link:.1e} m, 관절 한계 차 {e_lim:.1e}")
    # 역기구학 범위 (11/15 측정과 같은 격자: 종이 전체 7×7, 높이 4개, 손 돌림 3개 = 588점)
    from piper_kin import HOME_Q, TABLE_Z
    ok = 0
    n = 0
    for x in np.linspace(0.30 - 0.105, 0.30 + 0.105, 7):
        for y in np.linspace(-0.1485, 0.1485, 7):
            for z in (0.01, 0.05, 0.10, 0.15):
                for yaw in np.radians([-60, 0, 60]):
                    want = np.array([x, y, TABLE_Z + z])
                    q = HOME_Q.copy()
                    for _ in range(6):
                        q = kin.ik(want, grip_orientation(yaw), q)
                    n += 1
                    ok += np.linalg.norm(kin.fk(q)[0] - want) < 0.005
    print(f"역기구학: {n}점 중 {ok}점이 5 mm 안")
    assert e_pos < 1e-9 and e_mat < 1e-9 and e_jac < 1e-9 and e_link < 1e-9 and e_lim < 1e-12


if __name__ == "__main__":
    main()
