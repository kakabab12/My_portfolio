"""to_tinybc.py 시험 (11/26). zarr·imagecodecs·optree 가 있는 파이썬으로 돌린다:
  ~/smolVLA/chanhyuk_piper-teleop/.venv/bin/python test_tinybc.py
가짜 시범 두 개(가짜 로봇 카메라 사진)를 만들어 바꾸고, 석사형 tinybc (faster, 11/26 scripts/server.py) 와 같은
배열 경로·모양·chunks·shards·JPEG 95 인지, 값이 원래 기록과 같은지, episode_ends 가 맞는지 본다.
석사형 코드(data.py)로 직접 열어 보는 시험은 11/26 에 따로 했다 (일지)."""
import os
import shutil
import tempfile

import numpy as np

from recorder import FakeCam, jpeg
from to_tinybc import LEAVES, episode_rows, open_store, write_episode

FAILS = []


def check(name, ok):
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        FAILS.append(name)


def fake_episode(path, n, t0=0.0, with_cams=True, with_qvel=True):
    cams = {"wrist": FakeCam("wrist"), "front": FakeCam("top")}
    t = t0 + np.arange(n) * 0.01
    q = np.stack([np.sin(t + k) * 0.1 for k in range(6)], 1)
    rows = dict(t=t, q=q, grip=np.linspace(0.07, 0.02, n), q_cmd=q + 0.001, grip_cmd=np.linspace(0.07, 0.0, n),
                follow=np.r_[np.zeros(5, bool), np.ones(n - 10, bool), np.zeros(5, bool)])
    if with_qvel:
        rows["qvel"] = np.gradient(q, t, axis=0)
        rows["grip_force"] = np.full(n, 1.5)
    if with_cams:
        for k, c in cams.items():
            c.t0 -= 1.0                                            # 첫 장부터 그림이 바뀌게
            arr = np.empty(n, dtype=object)
            for i in range(n):
                c.t0 -= 0.034                                      # 줄마다 다음 장
                arr[i] = jpeg(c.latest(), 640, quality=95)
            rows[f"{k}_jpg"] = arr
    np.savez(path, **rows)


def main():
    import zarr
    tmp = tempfile.mkdtemp()
    try:
        fake_episode(os.path.join(tmp, "ep_000.npz"), 40)
        fake_episode(os.path.join(tmp, "ep_001.npz"), 30, t0=5.0, with_qvel=False)
        fake_episode(os.path.join(tmp, "ep_002.npz"), 20, with_cams=False)
        out = os.path.join(tmp, "t.zarr")
        g, n0 = open_store(out)
        check("새 store 는 빈 채로 시작 (episode_ends [])", n0 == 0 and list(g.attrs["episode_ends"]) == [])
        e0, why0 = episode_rows(os.path.join(tmp, "ep_000.npz"))
        e1, _ = episode_rows(os.path.join(tmp, "ep_001.npz"))
        e2, why2 = episode_rows(os.path.join(tmp, "ep_002.npz"))
        check("로봇 카메라가 없는 시범은 건너뜀", e2 is None and "카메라" in why2)
        check("앞뒤 따라가기 꺼진 줄 5개씩 자름 (40 → 30)", e0["info"]["rows"] == 30)
        check("11/26 전 기록 (qvel 없음) 은 차분으로", e1["info"]["qvel"] == "finite_difference")
        write_episode(g, e0)
        write_episode(g, e1)
        g = zarr.open_group(out)
        check("episode_ends = 누적 끝 [30, 50]", list(g.attrs["episode_ends"]) == [30, 50])
        want = {"actions": (50, 7), "info/dt": (50, 1), "obs/images/top": (50, 480, 640, 3),
                "obs/images/wrist": (50, 480, 640, 3), "obs/qpos": (50, 7), "obs/qvel": (50, 7)}
        shapes = {k: g.get_array("data/" + k).shape for k in LEAVES}
        check("배열 6개 경로·모양이 tinybc 와 같음", shapes == want)
        meta = {k: g.get_array("data/" + k).metadata.to_dict() for k in LEAVES}

        def grid(m, key):
            for c in m["codecs"]:
                if c["name"] == "sharding_indexed":
                    return tuple(m["chunk_grid"]["configuration"]["chunk_shape"]), tuple(c["configuration"]["chunk_shape"])
            return None
        img = meta["obs/images/top"]
        inner = [c for c in img["codecs"] if c["name"] == "sharding_indexed"][0]["configuration"]["codecs"]
        check("사진: shard 64장, chunk 1장, JPEG 95 (tinybc server.py 와 같음)",
              grid(img, "") == ((64, 480, 640, 3), (1, 480, 640, 3))
              and any(c["name"] == "imagecodecs_jpeg" and c["configuration"].get("level") == 95 for c in inner))
        check("관절: shard 1024줄, chunk 1줄 / dt: shard 8192, chunk 1024",
              grid(meta["obs/qpos"], "") == ((1024, 7), (1, 7)) and grid(meta["info/dt"], "") == ((8192, 1), (1024, 1)))
        d = np.load(os.path.join(tmp, "ep_000.npz"), allow_pickle=True)
        act = g.get_array("data/actions")[:30]
        qp = g.get_array("data/obs/qpos")[:30]
        qv = g.get_array("data/obs/qvel")[:30]
        check("actions = 보낸 관절 + 그리퍼 폭/0.1, qpos = 실제 관절 + 폭, qvel 7번째 = 그리퍼 힘",
              np.allclose(act[:, :6], d["q_cmd"][5:35]) and np.allclose(act[:, 6], d["grip_cmd"][5:35] / 0.1)
              and np.allclose(qp[:, :6], d["q"][5:35]) and np.allclose(qp[:, 6], d["grip"][5:35]) and np.allclose(qv[:, 6], 1.5))
        import cv2
        src = cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(d["front_jpg"][5]), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        got = g.get_array("data/obs/images/top")[0]
        psnr = 10 * np.log10(255 ** 2 / max(np.mean((src.astype(float) - got.astype(float)) ** 2), 1e-9))
        check(f"사진은 RGB 로, 원래와 거의 같음 (PSNR {psnr:.0f} dB > 35)", psnr > 35)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("결과:", "모두 통과" if not FAILS else f"{len(FAILS)}개 실패")
    return not FAILS


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
