"""헤드트래커 실시간 조절 UI — 볼륨 조절 같은 슬라이더로 커서 감도(화면 끝까지의 고개 각도)를
켜져 있는 트래커에 바로 반영한다 (2026-08-28 신설, 사용자 요청).

동작 방식 — 이 창과 트래커(eyebrow.py/forehead.py)는 서로 다른 프로세스라
직접 통신하지 않는다. 대신 아주 단순한 파일 하나로 이어준다:

    이 UI가 슬라이더를 움직일 때마다 --file로 받은 JSON에 즉시 저장
    -> 트래커가 매 프레임(정확히는 몇 프레임에 한 번, 부담 없도록) 그
       파일의 수정 시각을 확인해서 바뀌었으면 다시 읽어 반영

이 구조라서 자연히 요청한 두 조건을 만족한다:
  · "실시간 반영" — 트래커가 다음 확인 주기(수십 ms)에 알아서 읽어간다
  · "UI 꺼도 적용 유지" — 값은 파일에 남아 있으므로 이 창을 닫아도, 심지어
    트래커를 재시작해도 마지막 값 그대로 시작한다

새 의존성 없음 — Tkinter는 파이썬 표준 라이브러리에 포함돼 있다.

실행 (트래커 콘솔에서 "tune"+Enter를 치면 자동으로 이렇게 실행된다 —
직접 실행할 일은 거의 없다):
    py scripts/tuning_ui.py --tracker eyebrow --file eyebrow_tuning.json
"""
import argparse
import json
import os
import tkinter as tk
from tkinter import ttk

# 트래커별 기본값 — 파일이 아직 없을 때(처음 켤 때) 슬라이더 초기 위치로
# 쓴다. 각 트래커 .py의 ORIENTATION_HALF_SPAN_*_DEG와 맞춰 둔다.
#
# ★2026-09-24 감도·곡률 슬라이더 셋(sensitivity_x/y, arc_compensation)을 뺐다.
# 세 트래커 모두 상대 회전 매핑(head_orientation.py)으로 커서를 정하는데, 그 경로는
# 이 세 값을 쓰지 않는다 — 움직여도 커서가 전혀 바뀌지 않는 손잡이였다. 실제로
# 연구실 eyebrow_tuning.json에 arc_compensation -1.008이 저장돼 있었는데 효과가
# 없었다. 감도 손잡이는 "고개를 몇 도 돌리면 화면 끝인가" 하나다.
_DEFAULTS = {
    "eyebrow": {"orientation_half_span_x_deg": 15.0, "orientation_half_span_y_deg": 10.0},
    "forehead": {"orientation_half_span_x_deg": 15.0, "orientation_half_span_y_deg": 10.0},
    "head": {"orientation_half_span_x_deg": 15.0, "orientation_half_span_y_deg": 10.0},
}

# 각도 손잡이 범위 — 5도면 살짝만 돌려도 화면 끝(매우 민감), 40도면 크게
# 돌려야 한다(정밀). head_orientation이 60도에서 잘라내므로 그 안에 둔다
HALF_SPAN_RANGE = (5.0, 40.0)


def load_tuning(path, tracker):
    """파일이 있으면 그 값을, 없으면(처음 실행) 트래커 기본값을 돌려준다."""
    defaults = dict(_DEFAULTS[tracker])
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        defaults.update({k: v for k, v in saved.items() if k in defaults})
    except (ValueError, OSError):
        pass   # 손상된 파일 — 기본값으로 새로 시작
    return defaults


def save_tuning(path, values):
    # 트래커가 읽는 도중의 파일을 반쪽만 쓴 상태로 보게 하지 않으려고
    # 임시 파일에 쓴 뒤 통째로 바꿔치기한다(os.replace는 원자적이다)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


class TuningWindow:
    def __init__(self, root, path, tracker):
        self._path = path
        self._values = load_tuning(path, tracker)

        root.title(f"{tracker} 실시간 조절")
        root.attributes("-topmost", True)   # 트래커 카메라 창에 가려지지 않게

        self._sliders = {}
        self._add_slider(root, "orientation_half_span_x_deg", "가로 각도",
                         *HALF_SPAN_RANGE, row=0)
        self._add_slider(root, "orientation_half_span_y_deg", "세로 각도",
                         *HALF_SPAN_RANGE, row=1)

        # 안내 문구는 슬라이더 아래 줄에 둔다 — 예전엔 row=3이라 "가로 각도"
        # 슬라이더와 같은 줄에 겹쳐 그려졌다(2026-09-24 발견)
        note = tk.Label(
            root, fg="gray",
            text="이 창을 닫아도 방금 조절한 값은 그대로 적용돼 있습니다.")
        note.grid(row=2, column=0, columnspan=3, pady=(10, 4), padx=10)

    def _add_slider(self, root, key, label_text, lo, hi, row):
        tk.Label(root, text=label_text, width=10, anchor="w").grid(
            row=row, column=0, padx=(10, 4), pady=6)
        value_label = tk.Label(root, width=7, anchor="e")
        value_label.grid(row=row, column=2, padx=(4, 10))

        def _on_move(raw):
            value = float(raw)
            self._values[key] = value
            value_label.config(text=f"{value:.3f}")
            save_tuning(self._path, self._values)

        scale = ttk.Scale(root, from_=lo, to=hi, orient="horizontal", length=260,
                          command=_on_move)
        scale.set(self._values[key])
        scale.grid(row=row, column=1, padx=4, pady=6)
        value_label.config(text=f"{self._values[key]:.3f}")
        self._sliders[key] = scale


def main():
    parser = argparse.ArgumentParser(description="헤드트래커 실시간 조절 UI")
    parser.add_argument("--tracker", required=True, choices=sorted(_DEFAULTS.keys()))
    parser.add_argument("--file", required=True, help="트래커와 값을 주고받을 JSON 경로")
    args = parser.parse_args()

    root = tk.Tk()
    TuningWindow(root, args.file, args.tracker)
    root.mainloop()


if __name__ == "__main__":
    main()
