"""디버그 시각화 — 포즈·잠금 상태를 프레임 위에 그린다 (디버그 창에 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 머리 박스
TRACKED_COLOR = (60, 220, 60)    # 추적 손(단일 손 — 2026-07-31 라벨 제거)
CANDIDATE_COLOR = (150, 150, 150)  # 게이트 통과 후보 손 — 획득 전 상황 확인용
SHAPE_TAG = {"fist": "(F)", "finger": "(1)", "open": "(5)"}   # F=주먹, 1=한 손가락,
                                               # 5=손바닥(2026-07-31 temp 계층). 불명은 무표시
CURSOR_COLOR = (0, 220, 0)      # 헤드트래커 모드 커서(2026-07-30 병합)
MODE_COLOR = (255, 255, 0)
RECENTER_PROGRESS_COLOR = (0, 165, 255)  # calibration 응시 진행 링(2026-07-30) — 커서색과 구분


def draw_user_hands(frame, hand_selector):
    """추적 손 박스·추적점과 후보 손 점을 그린다.

    2026-07-31 단일 손 추적(라벨 제거): 좌/우 점 2개 대신 **추적 손 1점**(초록
    원 + 모양 태그)과 게이트 통과 후보(회색 점)를 표시 — 어떤 손이 잡혔고
    누가 후보로 대기 중인지 실기에서 바로 확인. 앵커 머리 상자("USER HEAD" —
    몸통판: 포즈 기반, 마스크·모자 무관)는 유지.
    """
    if getattr(hand_selector, "anchor_head_box", None) is not None:
        x1, y1, x2, y2 = hand_selector.anchor_head_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER HEAD", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    if hand_selector.locked_box is not None:
        x1, y1, x2, y2 = hand_selector.locked_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER HAND", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    for point in hand_selector.candidate_points():
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, CANDIDATE_COLOR, 1)
    signal = hand_selector.user_hand_signal()
    if signal is not None:
        shape, point = signal[0], signal[1]   # (모양, 좌표, 라벨, 검지비율) — 나머지는 시각화에 안 씀
        x_px, y_px = int(point[0]), int(point[1])
        cv2.circle(frame, (x_px, y_px), 10, TRACKED_COLOR, 2)
        cv2.putText(
            frame, "H" + SHAPE_TAG.get(shape, ""), (x_px + 12, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACKED_COLOR, 2,
        )
    return frame



def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-16).

    SCALE=어깨 스케일 / ARM=활성 팔+손 모양(원시 판별) / RET=복귀 삼킴 예약 방향 /
    SWIPE=진행도(±1.0 판정) / LATCH=고정 모양(F=주먹, 1=한 손가락, -=없음)과
    전환 후보(cand 모양:연속 수) — 판정은 래치만 본다 (2026-07-28 v3).
    TAP=탭 진단(2026-08-03 현장 대응 — "검지 까딱이 잘 안 잡힌다"):
    idx=검지 비율/하강선(검지 까딱 채널) · dy=최고점 대비 하강/임계(손목 까딱
    채널) · dips=누적 까딱 수. 까딱할 때 idx가 low 아래로 또는 dy가 임계 위로
    가면 dips가 오른다 — 안 오르면 그 채널 임계를 낮춘다.
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    swallow = debug.get("swallow")
    swallow_tag = f" [RET:{swallow}]" if swallow else ""
    side = debug.get("active_side") or "-"
    shape_tag = SHAPE_TAG.get(debug.get("hand_shape"), "")
    latch_label = {"fist": "F", "finger": "1", "open": "5"}.get(
        debug.get("latched_shape"), "-")
    candidate = debug.get("latch_candidate")
    latch_tag = latch_label + (f" cand:{candidate}" if candidate else "")
    lines = [
        f"SCALE {debug.get('body_scale', 0):.2f}  ARM {side}{shape_tag}{swallow_tag}",
        f"SWIPE x{debug.get('swipe_progress_x', 0):+.2f} y{debug.get('swipe_progress_y', 0):+.2f}"
        f"  LATCH {latch_tag}",
    ]
    index_ratio = debug.get("tap_index_ratio")
    if index_ratio is not None:
        # 까딱 순간 idx가 low 아래로 내려가면 dips가 오른다 — 안 내려가면 임계를 낮춘다
        low = debug.get("tap_low")
        lines.append(
            f"TAP idx {index_ratio:.2f}/low {low:.2f}"
            f"  dy {debug.get('tap_drop_y', 0):.2f}/{debug.get('tap_move_dip', 0):.2f}"
            f"  dips {debug.get('tap_dips', 0)}"
            if low is not None else f"TAP idx {index_ratio:.2f} (대기)")
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 24 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    return frame


def draw_cursor(frame, cursor_x_ratio, cursor_y_ratio, recenter_progress_ratio=0.0):
    """헤드트래커 모드 커서(코끝 매핑)를 화면 비율 좌표로 그린다 (2026-07-30 병합).

    recenter_progress_ratio(0~1)가 0보다 크면 커서 둘레에 진행 링을 함께 그린다 —
    calibration 응시가 몇 초 남았는지 숫자 없이 바로 보이게 한다(2026-07-30 사용자
    요청). 진행 표시를 커서 자리에 그리는 이유: 다른 자리(계기판 등)에 표시하면
    확인하려 시선을 옮기는 순간 커서가 반경을 벗어나 응시 자체가 끊긴다.
    """
    if cursor_x_ratio is None:
        return frame
    h_px, w_px = frame.shape[:2]
    x_px, y_px = int(cursor_x_ratio * w_px), int(cursor_y_ratio * h_px)
    cv2.circle(frame, (x_px, y_px), 14, CURSOR_COLOR, 2)
    cv2.drawMarker(frame, (x_px, y_px), CURSOR_COLOR, cv2.MARKER_CROSS, 10, 2)
    if recenter_progress_ratio > 0.0:
        end_angle_deg = 360.0 * recenter_progress_ratio
        cv2.ellipse(frame, (x_px, y_px), (22, 22), -90, 0, end_angle_deg,
                    RECENTER_PROGRESS_COLOR, 3)
    return frame


def draw_head_debug_panel(frame, debug):
    """헤드트래커 계기판 — hand 모드의 draw_debug_panel과 자리를 공유한다(2026-07-30).

    JAW=입벌림 점수/기준선 · EYE=눈감김 진행도 · DWELL=응시(select) 진행도 ·
    RECENTER=응시(calibration) 진행도 · PUCKER=입오므림(기본 비활성 — 계기판 참고용).
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    jaw_base = debug.get("jaw_base")
    eye_base = debug.get("eye_base")
    pucker_base = debug.get("pucker_base")
    lines = [
        f"JAW {debug.get('jaw_open', 0):.2f}/{'-' if jaw_base is None else f'{jaw_base:.2f}'}"
        f"  EYE {debug.get('eye_close', 0):.2f}/{'-' if eye_base is None else f'{eye_base:.2f}'}"
        f" ({debug.get('eye_progress', 0):.0%})",
        f"DWELL {debug.get('dwell_progress', 0):.0%}"
        f"  RECENTER {debug.get('recenter_progress', 0):.0%}"
        f"  PUCKER {debug.get('pucker', 0):.2f}/{'-' if pucker_base is None else f'{pucker_base:.2f}'}",
    ]
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 24 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    return frame


def draw_mode_banner(frame, input_mode):
    """지금 입력 모드(손/머리)를 상단 중앙에 표시한다 (2026-07-30 병합 — 전환 확인용)."""
    label = "HAND MODE" if input_mode == "hand" else "HEAD MODE"
    w_px = frame.shape[1]
    cv2.putText(frame, label, (w_px // 2 - 70, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, MODE_COLOR, 2)
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
