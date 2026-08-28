"""카메라 한 장으로 [원본 | 깊이 | 조명] 3단 비교 이미지를 만든다.

결과가 실제로 어떻게 보이는지 남기는 용도. 카메라가 없으면 --image로 사진 지정.
"""
import argparse
import time

import cv2
import numpy as np

from depth_lighting import DepthLighting
from depth_model import MidasDepth


def label(img, text):
    """이미지 위쪽에 검은 띠를 깔고 제목을 적는다."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--image", default=None, help="카메라 대신 쓸 사진 경로")
    ap.add_argument("--out", default="sample_output.png")
    ap.add_argument("--warmup", type=int, default=12, help="카메라 자동노출 안정화용 버림 프레임")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("사진을 읽지 못했습니다:", args.image)
            return 1
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("카메라를 열지 못했습니다.")
            return 1
        frame = None
        for _ in range(args.warmup):        # 첫 프레임들은 노출이 안 잡혀 어둡다
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.05)
        cap.release()
        if frame is None:
            print("프레임을 받지 못했습니다.")
            return 1
        frame = cv2.flip(frame, 1)

    model = MidasDepth()
    t = time.perf_counter()
    depth = model.infer(frame)
    infer_ms = (time.perf_counter() - t) * 1000

    lighting = DepthLighting()
    small = cv2.resize(depth, (128, 128), interpolation=cv2.INTER_LINEAR)
    t = time.perf_counter()
    lit = lighting.shade_frame(frame, small, (0.45, -0.10, 0.12))
    light_ms = (time.perf_counter() - t) * 1000

    h, w = frame.shape[:2]
    depth_vis = cv2.applyColorMap(
        cv2.resize((depth * 255).astype(np.uint8), (w, h)), cv2.COLORMAP_INFERNO)

    combo = np.hstack([
        label(frame, "1. camera input"),
        label(depth_vis, "2. AI depth (%.0f ms)" % infer_ms),
        label(lit, "3. depth-reactive light (%.0f ms)" % light_ms)])
    ok, buf = cv2.imencode(".png", combo)
    if ok:
        open(args.out, "wb").write(buf.tobytes())     # 한글 경로 대응(cv2.imwrite는 실패한다)
        print("저장: %s" % args.out)
    print("깊이 추정 %.0fms + 조명 %.0fms = %.0fms (%.1f fps 상당)"
          % (infer_ms, light_ms, infer_ms + light_ms, 1000 / (infer_ms + light_ms)))
    print("깊이 범위 %.2f~%.2f (1=가까움)" % (depth.min(), depth.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
