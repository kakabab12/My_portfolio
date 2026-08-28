"""디버그 시각화 — 얼굴 잠금·커서 상태를 프레임 위에 그린다 (예시 UI 스트림에도 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 얼굴 박스
CURSOR_COLOR = (80, 220, 120)    # 코끝 커서 점
CURSOR_OUTLINE_COLOR = (25, 30, 25)   # 커서 외곽 테두리 — 밝은 배경에서도 보이게
CURSOR_RADIUS_RATIO = 0.022      # 커서 반지름(프레임 높이 대비) — 해상도가 바뀌어도 같은 비율


def draw_person_lock(frame, person_lock, cursor_x_ratio, cursor_y_ratio):
    """잠긴 사용자의 얼굴 박스와 코끝 커서 위치를 그린다.

    /data 폴링(250ms)만으로 커서를 그리면 끊겨 보여 "포인팅"처럼 느껴지지 않는다 —
    비디오 프레임에 서버 사이드로 직접 그려 캡처 FPS 그대로의 부드러운 커서를 보여준다
    (2026-07-18 헤드트래커 전환).
    """
    if person_lock.locked_face is not None:
        x1, y1, x2, y2 = (int(v) for v in person_lock.locked_face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER LOCK", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    if cursor_x_ratio is not None:
        h_px, w_px = frame.shape[:2]
        x_px, y_px = int(cursor_x_ratio * w_px), int(cursor_y_ratio * h_px)
        radius_px = max(10, int(h_px * CURSOR_RADIUS_RATIO))
        cv2.circle(frame, (x_px, y_px), radius_px + 2, CURSOR_OUTLINE_COLOR, 6)
        cv2.circle(frame, (x_px, y_px), radius_px, CURSOR_COLOR, 3)
        cv2.circle(frame, (x_px, y_px), 4, CURSOR_COLOR, -1)
    return frame



def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-18 헤드트래커 전환).

    JAW/EYE=현재 점수(괄호 안은 2026-07-20 도입한 평상시 기준선) / DWELL·EYE_HOLD=
    각각 진행도(1.0 도달 시 확정).
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    cursor_tag = (
        f"({debug.get('cursor_x_ratio')},{debug.get('cursor_y_ratio')})"
        if debug.get("cursor_x_ratio") is not None else "미추적"
    )
    jaw_baseline = debug.get("jaw_baseline")
    eye_baseline = debug.get("eye_baseline")
    pucker_baseline = debug.get("mouth_pucker_baseline")
    if jaw_baseline is None:
        jaw_line = "JAW -(캘리브레이션 중)"
    else:
        jaw_line = f"JAW {debug.get('jaw_open_score', 0):.2f}(base {jaw_baseline:.2f})"
    if eye_baseline is None:
        eye_line = "EYE -(캘리브레이션 중)"
    else:
        eye_line = (
            f"EYE {debug.get('eye_close_score', 0):.2f}(base {eye_baseline:.2f})"
            f"  DWELL {debug.get('dwell_progress_ratio', 0):.2f}"
            f"  EYE_HOLD {debug.get('eye_close_progress_ratio', 0):.2f}"
        )
    if pucker_baseline is None:
        pucker_line = "PUCKER -(캘리브레이션 중)"
    else:
        pucker_line = f"PUCKER {debug.get('mouth_pucker_score', 0):.2f}(base {pucker_baseline:.2f})"
    lines = [f"CURSOR {cursor_tag}", jaw_line, eye_line, pucker_line]
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 24 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    return frame


def draw_status(frame, avg_fps, gesture_event=None):
    """FPS와 최근 확정 이벤트를 좌상단에 표시한다."""
    cv2.putText(
        frame, f"FPS {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2
    )
    if gesture_event is not None:
        cv2.putText(
            frame,
            f"EVENT {gesture_event.class_name}",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            EVENT_COLOR,
            2,
        )
    return frame
