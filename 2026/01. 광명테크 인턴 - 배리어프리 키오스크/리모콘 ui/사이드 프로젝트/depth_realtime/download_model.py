"""깊이 추정 모델(MiDaS v2.1 small)을 models/에 내려받는다.

받은 뒤 크기와 실제 로딩까지 확인한다 — 중간에 끊긴 파일이 남으면 나중에
엉뚱한 오류로 나타나서 원인을 찾기 어렵다(실제로 이 폴더에 128KB짜리 잘린
파일이 한 번 생겼었다).
"""
import os
import sys
import urllib.request

URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "weights", "midas_v21_small.onnx")
EXPECTED_BYTES = 66_764_249      # 실측값 — 이보다 작으면 받다 만 파일이다


def _progress(done, total):
    if total <= 0:
        return
    pct = done * 100 // total
    bar = "#" * (pct // 4)
    sys.stdout.write(f"\r  [{bar:<25}] {pct:3d}%  {done/1e6:5.1f}/{total/1e6:.1f} MB")
    sys.stdout.flush()


def download(force=False):
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and not force:
        size = os.path.getsize(DEST)
        if size == EXPECTED_BYTES:
            print(f"이미 있습니다: {DEST} ({size/1e6:.1f} MB)")
            return DEST
        print(f"크기가 다릅니다({size/1e6:.1f} MB, 정상 {EXPECTED_BYTES/1e6:.1f} MB) — 다시 받습니다")

    print(f"내려받는 중: {URL}")
    tmp = DEST + ".part"        # 다 받기 전에는 최종 이름을 안 쓴다 — 끊겨도 잘린 파일이 안 남는다
    try:
        with urllib.request.urlopen(URL, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
        print()
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"실패: {type(e).__name__}: {e}")
        print("직접 받아서 models/midas_v21_small.onnx 로 저장해도 됩니다:")
        print(" ", URL)
        return None

    size = os.path.getsize(tmp)
    if size < EXPECTED_BYTES * 0.9:
        os.remove(tmp)
        print(f"받은 크기가 너무 작습니다({size/1e6:.1f} MB) — 삭제했습니다.")
        return None
    os.replace(tmp, DEST)
    print(f"저장 완료: {DEST} ({size/1e6:.1f} MB)")
    return DEST


if __name__ == "__main__":
    path = download(force="--force" in sys.argv)
    if path is None:
        raise SystemExit(1)
    # 실제로 열리는지까지 확인 — 파일 크기만 맞고 내용이 깨진 경우를 걸러낸다
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        print(f"모델 확인 OK — 입력 {i.name} {i.shape}")
    except Exception as e:
        print(f"파일은 받았지만 모델을 열지 못했습니다: {type(e).__name__}: {e}")
        raise SystemExit(1)
