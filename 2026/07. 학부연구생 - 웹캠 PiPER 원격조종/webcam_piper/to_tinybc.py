"""우리 시범 기록(ep_XXX.npz)을 석사형 tinybc 형식(zarr)으로 바꾼다 (2026-11-26).

석사형 VR 조종(github chanhyuk-jung/tinybc, faster 브랜치 scripts/server.py)이 쓰는 저장 모양과 똑같이 만든다.
그래서 VR 시범과 웹캠 시범을 같은 학습 코드에 섞어 넣을 수 있다.

  data/actions            (N, 7)  보낸 관절 목표 6 + 트리거 (= 보낸 그리퍼 폭 / 0.1 m, tinybc 는 폭 = 트리거 × 0.1)
  data/obs/qpos           (N, 7)  실제 관절 6 (rad) + 실제 그리퍼 폭 (m)
  data/obs/qvel           (N, 7)  관절 속도 6 (rad/s) + 그리퍼 힘
  data/obs/images/wrist   (N, 480, 640, 3) uint8 RGB, JPEG 95
  data/obs/images/top     (N, 480, 640, 3) uint8 RGB, JPEG 95   (우리 기록의 front 카메라)
  data/info/dt            (N, 1)  줄 사이 시간 (s)
  그룹 속성: treespec (optree 구조, base64 pickle), episode_ends (시범마다 끝 줄 번호, 누적)

실행 (zarr·imagecodecs·optree 가 필요하다. 찬혁님 piper-teleop .venv 에 있어서 그것으로 돌린다):
  ~/smolVLA/chanhyuk_piper-teleop/.venv/bin/python to_tinybc.py data/20261126_153000 --out data/tinybc_webcam.zarr
  (폴더나 ep_XXX.npz 를 여러 개 줘도 됨. --out 이 이미 있으면 뒤에 붙인다)

거르는 것
  - 로봇 쪽 카메라(손목 wrist, 위 front/top) 사진이 둘 다 없는 시범은 건너뛴다 (tinybc 는 사진이 꼭 있어야 함).
    robot_server 를 --wrist-cam N --top-cam M 으로 켜고 기록해야 한다. --blank-missing 은 검은 그림으로 채움 (길 시험용만).
  - 앞뒤에 따라가기가 꺼져 있던 줄은 자른다 (tinybc 는 시작(squeeze) 뒤부터 기록). --keep-idle 이면 그대로.
  - 11/26 전 기록은 관절 속도가 없어서 실제 관절의 차분으로 채우고 그리퍼 힘은 0 으로 둔다 (속성 webcam_piper 에 적음).

석사형 코드(11/26 faster)를 읽어 보면 server.py 가 mark() 를 부르지 않고, mark() 도 episode_ends 가 비어 있으면 아무것도 안 넣어서
석사형 서버가 쓴 store 는 episode_ends 가 [] 로 남을 것으로 보인다 (실제 store 로는 확인 안 함). 여기서는 시범을 넣을 때마다
누적 끝 번호를 넣는다. 섞이지 않게 웹캠 시범은 따로 store 를 쓰기를 권한다.
"""
import argparse
import base64
import glob
import json
import os
import pickle

import cv2
import numpy as np

W, H = 640, 480
GRIP_SCALE = 0.1                     # tinybc: 그리퍼 폭 (m) = 트리거 × 0.1


def spec():
    """tinybc scripts/server.py (faster, 11/26) 의 zarr_store.create(...) 와 같은 배열들. 열쇠 이름 순서는 optree 가 정렬."""
    from imagecodecs.zarr import Jpeg
    vec = dict(dtype=np.float64, chunks=(1, 7), shards=(1024, 7), compressors=None, serializer="auto")
    img = dict(dtype=np.uint8, chunks=(1, H, W, 3), shards=(64, H, W, 3), compressors=None)
    return {
        "actions": dict(shape=(0, 7), **vec),
        "info": {"dt": dict(shape=(0, 1), dtype=np.float64, chunks=(1024, 1), shards=(8192, 1), compressors=None, serializer="auto")},
        "obs": {
            "images": {"top": dict(shape=(0, H, W, 3), serializer=Jpeg(level=95), **img),
                       "wrist": dict(shape=(0, H, W, 3), serializer=Jpeg(level=95), **img)},
            "qpos": dict(shape=(0, 7), **vec),
            "qvel": dict(shape=(0, 7), **vec),
        },
    }


LEAVES = ["actions", "info/dt", "obs/images/top", "obs/images/wrist", "obs/qpos", "obs/qvel"]   # optree 정렬 순서


def open_store(path):
    """tinybc ZarrStore 와 같은 모양의 store 를 열거나 만든다. 돌려줌 (zarr group, 지금 줄 수)."""
    import optree
    import zarr
    from imagecodecs.zarr import register_codecs
    register_codecs()                                                   # tinybc data.py 처럼 JPEG 등 imagecodecs 압축을 zarr 에 등록
    g = zarr.group(path)
    if "treespec" not in g.attrs:
        g.update_attributes({"episode_ends": []})
        tree = spec()
        leaves, treespec = optree.tree_flatten(tree, is_leaf=lambda x: isinstance(x, dict) and "dtype" in x)
        names = ["data/" + "/".join(str(k) for k in p) for p in optree.treespec_paths(treespec)]
        assert [n[5:] for n in names] == LEAVES, names
        for name, kw in zip(names, leaves):
            g.create_array(name, **kw)
        g.update_attributes({"treespec": base64.b64encode(pickle.dumps(treespec)).decode("ascii")})
    n = g.get_array("data/actions").shape[0]
    ends = list(g.attrs.get("episode_ends", []))
    if ends and ends[-1] != n:
        print(f"[주의] store 의 episode_ends 끝 {ends[-1]} 과 실제 줄 수 {n} 이 다름 — 그대로 이어 붙임")
    elif not ends and n > 0:
        print(f"[주의] store 에 시범 경계 없이 {n} 줄이 이미 있음 (석사형 서버가 쓴 store 로 보임). 따로 store 를 쓰기를 권함")
    return g, n


def decode(buf, blank):
    if buf is None or len(buf) == 0:
        return np.zeros((H, W, 3), np.uint8) if blank else None
    img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((H, W, 3), np.uint8) if blank else None
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                         # tinybc CameraThread 는 RGB 로 저장


def episode_rows(path, keep_idle=False, blank_missing=False):
    """ep_XXX.npz 하나 → tinybc 한 시범 (dict of arrays) 또는 (None, 이유)."""
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    have = {k: f"{k}_jpg" in keys for k in ("wrist", "front")}
    if not all(have.values()) and not blank_missing:
        miss = [("wrist" if k == "wrist" else "top(front)") for k, v in have.items() if not v]
        return None, "로봇 카메라 사진 없음: " + ", ".join(miss)
    t = np.asarray(d["t"], np.float64)
    n = len(t)
    idx = np.arange(n)
    if not keep_idle and "follow" in keys:
        fol = np.asarray(d["follow"], bool)
        if not fol.any():
            return None, "따라가기를 한 줄이 없음"
        on = np.flatnonzero(fol)
        idx = np.arange(on[0], on[-1] + 1)
    q = np.asarray(d["q"], np.float64)[idx]
    grip = np.asarray(d["grip"], np.float64)[idx]
    q_cmd = np.asarray(d["q_cmd"], np.float64)[idx]
    grip_cmd = np.asarray(d["grip_cmd"], np.float64)[idx]
    grip = np.where(np.isfinite(grip), grip, grip_cmd)                  # 그리퍼를 못 읽은 줄은 보낸 폭으로
    tt = t[idx]
    dt = np.diff(tt, prepend=tt[0])
    dt[0] = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.01
    if "qvel" in keys:
        qd = np.asarray(d["qvel"], np.float64)[idx]
        force = np.asarray(d["grip_force"], np.float64)[idx] if "grip_force" in keys else np.zeros(len(idx))
        force = np.where(np.isfinite(force), force, 0.0)
        vel_src = "measured"
    else:
        qd = np.gradient(q, tt, axis=0) if len(tt) > 1 else np.zeros_like(q)
        force = np.zeros(len(idx))
        vel_src = "finite_difference"
    out = {
        "actions": np.column_stack([q_cmd, grip_cmd / GRIP_SCALE]),
        "dt": dt[:, None],
        "qpos": np.column_stack([q, grip]),
        "qvel": np.column_stack([qd, force]),
        "wrist": d["wrist_jpg"][idx] if have["wrist"] else [b""] * len(idx),
        "top": d["front_jpg"][idx] if have["front"] else [b""] * len(idx),
        "info": {"file": os.path.abspath(path), "rows": int(len(idx)), "rows_raw": int(n), "qvel": vel_src,
                 "hz": float(1.0 / np.median(dt)) if np.median(dt) > 0 else None,
                 "blank_images": [k for k, v in have.items() if not v]},
    }
    return out, None


def write_episode(g, ep, batch=256, blank_missing=False):
    """tinybc ZarrStore.write_episode 처럼 배치로 이어 붙이고, episode_ends 에 누적 끝을 넣는다."""
    arrs = {name: g.get_array("data/" + name) for name in LEAVES}
    n = len(ep["actions"])
    for s in range(0, n, batch):
        e = min(n, s + batch)
        arrs["actions"].append(ep["actions"][s:e].astype(np.float64))
        arrs["info/dt"].append(ep["dt"][s:e].astype(np.float64))
        arrs["obs/qpos"].append(ep["qpos"][s:e].astype(np.float64))
        arrs["obs/qvel"].append(ep["qvel"][s:e].astype(np.float64))
        for cam, key in (("top", "obs/images/top"), ("wrist", "obs/images/wrist")):
            imgs = [decode(b, blank_missing) for b in ep[cam][s:e]]
            arrs[key].append(np.stack(imgs).astype(np.uint8))
    total = arrs["actions"].shape[0]
    ends = list(g.attrs.get("episode_ends", []))
    ends.append(int(total))
    src = list(g.attrs.get("webcam_piper", []))
    src.append(ep["info"])
    g.update_attributes({"episode_ends": ends, "webcam_piper": src})
    return total


def main():
    ap = argparse.ArgumentParser(description="webcam_piper 기록(ep_XXX.npz) → tinybc zarr")
    ap.add_argument("inputs", nargs="+", help="기록 폴더(data/날짜_시각) 나 ep_XXX.npz")
    ap.add_argument("--out", required=True, help="tinybc zarr store 경로 (없으면 만듦, 있으면 뒤에 붙임)")
    ap.add_argument("--keep-idle", action="store_true", help="따라가기가 꺼진 앞뒤 줄도 그대로 둠")
    ap.add_argument("--blank-missing", action="store_true", help="로봇 카메라 사진이 없으면 검은 그림으로 채움 (길 시험용만)")
    a = ap.parse_args()
    files = []
    for p in a.inputs:
        files += sorted(glob.glob(os.path.join(p, "ep_*.npz"))) if os.path.isdir(p) else [p]
    if not files:
        raise SystemExit("ep_*.npz 를 못 찾음")
    g, n0 = open_store(a.out)
    done = 0
    for f in files:
        ep, why = episode_rows(f, a.keep_idle, a.blank_missing)
        if ep is None:
            print(f"[건너뜀] {f}: {why}")
            continue
        total = write_episode(g, ep, blank_missing=a.blank_missing)
        done += 1
        print(f"[넣음] {f}: {ep['info']['rows']}줄 (원래 {ep['info']['rows_raw']}줄, 약 {ep['info']['hz']:.0f} Hz, 관절 속도 {ep['info']['qvel']})"
              f" → 누적 {total}줄")
    g2 = g.attrs.asdict() if hasattr(g.attrs, "asdict") else dict(g.attrs)
    print(f"[끝] {done}/{len(files)}개 시범을 {a.out} 에 넣음. 줄 {n0} → {g.get_array('data/actions').shape[0]}, "
          f"episode_ends {json.dumps(g2.get('episode_ends'))}")


if __name__ == "__main__":
    main()
