"""ui 모듈 — 로터 리모컨 오버레이 창 (tkinter, 2026-08-05 신설 — feat/rotor_remote 판).

요구(사용자 2026-08-05): 카메라 화면 위 그리기가 아니라 **윈도우 데스크톱 창**으로
①항상 위(topmost) ②배경 투명 ③화면 고정 배치 — 수정이 쉬워야 한다.
- tkinter는 파이썬 내장 — 추가 설치 없음 (PyQt 대비 의존 0 — 사용자 선택).
- 배치·크기·색·버튼은 전부 config rotor / rotor.ui에서 읽는다(기획서 4.7) —
  모니터가 바뀌면 x_px/y_px만, 버튼을 늘리려면 rotor.buttons에 줄만 추가.
- 투명: overrideredirect(테두리 제거) + attributes("-transparentcolor", 키색).
  캔버스 바탕을 키색으로 칠하면 그 색 픽셀이 통째로 뚫려 버튼·바늘만 화면에
  뜬다 (윈도우 전용 속성 — 이 프로젝트의 타깃 OS는 windows 고정, config runtime).
- 항상 위: attributes("-topmost", True) + 주기적 lift() — topmost 창끼리는
  마지막에 설정한 쪽이 이기므로 topmost_lift_ms마다 탈환한다.
- tkinter는 스레드 안전하지 않다 — **모든 tk 호출을 이 스레드 하나에** 가둔다:
  창 생성·after 갱신·mainloop 전부 run() 안. 파이프라인(추론 스레드)과는
  RotorController.snapshot()(락 보호 사본)으로만 만난다.
- 창 표시 자체가 로터 on/off를 따라간다: 꺼짐 = withdraw(숨김), 켜짐 = deiconify.

★2026-08-05 래칫 다이얼(사용자 확인 8차 — 기아 자동차 다이얼 방식): 버튼은
**제자리 고정**(12시부터 시계방향 균등 배치), **선택 원(orbit ring)**이 현재
포커스 버튼에 걸려 있고 래칫 스텝마다 옆 칸으로 점프한다("무엇이 선택되고
있는지" — 7차 요구 유지). 중앙 노브는 펼친 손의 **비틀림 각(twist)**을
그대로 보여준다 — 마커가 손을 따라 돌고, 양옆의 눈금(±step_threshold_deg)에
닿으면 찰칵 한 칸이다. 확정은 주먹(반짝임으로만 표시 — 차징 링 없음).
버튼 수는 config rotor.buttons 목록 길이(4개→9개 확장 대비)이고, 많아져
간격이 좁아지면 버튼 반지름을 자동으로 줄인다.
"""
import math
import threading

from src.utils.logger import get_logger

logger = get_logger("ui")


class RotorWindow(threading.Thread):
    """로터 리모컨 오버레이 — 데몬 스레드 하나가 tk 창을 소유한다."""

    def __init__(self, config, rotor):
        super().__init__(daemon=True, name="rotor_window")
        rotor_cfg = config["rotor"]
        ui_cfg = rotor_cfg["ui"]
        self._rotor = rotor
        # 버튼 목록·배치각은 판정(rotor.py)과 같은 규칙으로 계산한다 — 화면과
        # 판정이 같은 것을 가리켜야 한다 (12시부터 시계방향 균등 간격)
        self._buttons = [dict(button) for button in rotor_cfg["buttons"]]
        self._spacing_deg = 360.0 / len(self._buttons)
        # 래칫 눈금 표시는 판정과 같은 값을 그린다 — 화면과 판정이 같은 문턱을
        # 가리켜야 한다 (마커가 눈금에 닿으면 실제로 한 칸 넘어간다)
        self._step_threshold_deg = rotor_cfg["step_threshold_deg"]
        self._x_px = ui_cfg["x_px"]
        self._y_px = ui_cfg["y_px"]
        self._width_px = ui_cfg["width_px"]
        self._height_px = ui_cfg["height_px"]
        self._refresh_ms = ui_cfg["refresh_ms"]
        self._topmost_lift_ms = ui_cfg["topmost_lift_ms"]
        self._button_offset_ratio = ui_cfg["button_offset_ratio"]
        self._button_radius_ratio = ui_cfg["button_radius_ratio"]
        self._dial_radius_ratio = ui_cfg["dial_radius_ratio"]
        self._label_font_px = ui_cfg["label_font_px"]
        self._colors = dict(ui_cfg["colors"])
        self._root = None
        self._canvas = None
        self._is_shown = False
        self._lift_elapsed_ms = 0

    def run(self):
        """UI 스레드 본체 — 창 생성부터 mainloop까지 tk 호출 전부 여기서만."""
        import tkinter as tk   # 지연 임포트 — UI를 끈 환경은 tkinter 비용을 지지 않는다

        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)   # 테두리·제목줄 제거 — 고정 배치 오버레이
            # 화면 밖 배치 보정(2026-08-05 실기 — 사용자 보고 "창이 안 뜸"): config
            # 위치가 이 모니터 밖이면(1280x800 화면에 x=1520) 창이 멀쩡히 떠도
            # 안 보인다 — 로터·이벤트는 다 돌아 원인 파악이 어려웠다. 화면 안으로
            # 잘라 넣고 WARNING으로 알린다 (config 값 자체는 안 고친다)
            screen_w_px = self._root.winfo_screenwidth()
            screen_h_px = self._root.winfo_screenheight()
            x_px = min(max(self._x_px, 0), max(0, screen_w_px - self._width_px))
            y_px = min(max(self._y_px, 0), max(0, screen_h_px - self._height_px))
            if (x_px, y_px) != (self._x_px, self._y_px):
                logger.warning("로터 UI 창 위치 보정: config (%d,%d) -> (%d,%d) — "
                               "화면(%dx%d) 밖이라 안 보였을 위치. config rotor.ui의 "
                               "x_px/y_px를 이 모니터에 맞게 고칠 것",
                               self._x_px, self._y_px, x_px, y_px,
                               screen_w_px, screen_h_px)
            self._root.geometry(f"{self._width_px}x{self._height_px}"
                                f"+{x_px}+{y_px}")
            self._root.attributes("-topmost", True)
            key_color = self._colors["transparent_key"]
            self._root.attributes("-transparentcolor", key_color)
            self._root.configure(bg=key_color)
            self._canvas = tk.Canvas(self._root, width=self._width_px,
                                     height=self._height_px, bg=key_color,
                                     highlightthickness=0)
            self._canvas.pack()
            self._root.withdraw()   # 시작은 숨김 — 로터가 켜질 때 나타난다
            self._root.after(self._refresh_ms, self._refresh)
            logger.info("로터 UI 창 시작 (%dx%d @ %d,%d · 버튼 %d개)", self._width_px,
                        self._height_px, x_px, y_px, len(self._buttons))
            self._root.mainloop()
        except Exception:   # noqa: BLE001 — UI 실패가 엔진(판정·이벤트)을 못 죽이게
            logger.exception("로터 UI 창 실패 — 엔진은 계속 구동한다 (창만 없음)")

    # ----- 주기 갱신 (tk after 루프) -----

    def _refresh(self):
        import tkinter as tk

        try:
            snapshot = self._rotor.snapshot()
            self._sync_visibility(snapshot["is_on"])
            if self._is_shown:
                self._draw(snapshot)
                self._lift_elapsed_ms += self._refresh_ms
                if self._lift_elapsed_ms >= self._topmost_lift_ms:
                    self._lift_elapsed_ms = 0
                    self._root.lift()
                    self._root.attributes("-topmost", True)   # topmost 경쟁 탈환
            self._root.after(self._refresh_ms, self._refresh)
        except tk.TclError:
            pass   # 창이 닫히는 중 — 루프만 조용히 끝낸다 (데몬 스레드라 함께 종료)

    def _sync_visibility(self, is_rotor_on):
        if is_rotor_on and not self._is_shown:
            self._root.deiconify()
            self._root.lift()
            self._root.attributes("-topmost", True)
            self._is_shown = True
        elif not is_rotor_on and self._is_shown:
            self._root.withdraw()
            self._is_shown = False

    @staticmethod
    def _direction(angle_deg):
        """방향표 각도(0=12시·시계방향+) -> 화면 단위 벡터 (화면 y는 아래로 증가)."""
        angle_rad = math.radians(angle_deg)
        return math.sin(angle_rad), -math.cos(angle_rad)

    def _draw(self, snapshot):
        """스냅숏 1건을 화면으로 — 고정 버튼 + 회전 노브(방향표) + 유지 진행 링."""
        import tkinter as tk

        canvas = self._canvas
        canvas.delete("all")
        center_x = self._width_px / 2.0
        center_y = self._height_px / 2.0
        scale_px = min(self._width_px, self._height_px)
        offset_px = self._button_offset_ratio * scale_px
        # 버튼이 많아 간격이 좁아지면 반지름을 자동 축소 — 이웃 중심 간 거리를
        # 넘지 않게: 9개 확장 시 config 수정 없이 겹침 방지
        neighbor_gap_px = 2.0 * offset_px * math.sin(math.pi / len(self._buttons))
        radius_px = min(self._button_radius_ratio * scale_px, 0.45 * neighbor_gap_px)

        # 버튼 — 12시부터 시계방향 고정 배치 (돌지 않는다 — 사용자 확인 5차)
        for button_idx, button in enumerate(self._buttons):
            dir_x, dir_y = self._direction(button_idx * self._spacing_deg)
            zone_x = center_x + offset_px * dir_x
            zone_y = center_y + offset_px * dir_y
            fill = self._colors["button"]
            if button["name"] == snapshot.get("focus_name"):
                fill = self._colors["button_focus"]   # 방향표가 가리키는 버튼
            if button["name"] == snapshot.get("flash_name"):
                fill = self._colors["button_flash"]
            canvas.create_oval(zone_x - radius_px, zone_y - radius_px,
                               zone_x + radius_px, zone_y + radius_px,
                               fill=fill, outline=self._colors["button_border"],
                               width=2)
            canvas.create_text(zone_x, zone_y, text=button["label"],
                               fill=self._colors["label_text"],
                               font=("Segoe UI", -self._label_font_px, "bold"))

        # 선택 원(orbit ring — 사용자 확인 7차 유지): 현재 포커스 버튼에 걸린 원 —
        # 래칫 스텝마다 옆 칸으로 점프한다. "지금 쥐면 이게 나간다"의 표시
        focus_name = snapshot.get("focus_name")
        for button_idx, button in enumerate(self._buttons):
            if button["name"] != focus_name:
                continue
            dir_x, dir_y = self._direction(button_idx * self._spacing_deg)
            ring_x = center_x + offset_px * dir_x
            ring_y = center_y + offset_px * dir_y
            ring_px = radius_px + 6
            canvas.create_oval(ring_x - ring_px, ring_y - ring_px,
                               ring_x + ring_px, ring_y + ring_px,
                               outline=self._colors["progress"], width=4)

        # 중앙 노브 — 펼친 손 비틀림(twist)을 그대로 보여주는 손잡이:
        # 마커가 손을 따라 돌고, 양옆 눈금(±step_threshold_deg)에 닿으면 찰칵
        # 한 칸이다. 비틀림 None(펼친 손 아님·대기)이면 노브·눈금만 남는다
        knob_px = self._dial_radius_ratio * scale_px
        canvas.create_oval(center_x - knob_px, center_y - knob_px,
                           center_x + knob_px, center_y + knob_px,
                           fill=self._colors["button"],
                           outline=self._colors["button_border"], width=3)
        for tick_deg in (-self._step_threshold_deg, self._step_threshold_deg):
            dir_x, dir_y = self._direction(tick_deg)
            canvas.create_line(center_x + knob_px * 0.78 * dir_x,
                               center_y + knob_px * 0.78 * dir_y,
                               center_x + knob_px * 1.05 * dir_x,
                               center_y + knob_px * 1.05 * dir_y,
                               fill=self._colors["label_text"], width=3)
        twist_deg = snapshot.get("twist_deg")
        if twist_deg is not None:
            dir_x, dir_y = self._direction(twist_deg)
            marker_x = center_x + knob_px * 0.72 * dir_x
            marker_y = center_y + knob_px * 0.72 * dir_y
            canvas.create_line(center_x, center_y, marker_x, marker_y,
                               fill=self._colors["dial"], width=4)
            marker_r_px = max(4.0, knob_px * 0.16)
            canvas.create_oval(marker_x - marker_r_px, marker_y - marker_r_px,
                               marker_x + marker_r_px, marker_y + marker_r_px,
                               fill=self._colors["dial"], outline="")
