"""디버그 시각화 — 포즈·잠금 상태를 프레임 위에 그린다 (디버그 창에 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 머리 박스
TRACKED_COLOR = (60, 220, 60)    # 추적 손(단일 손 — 2026-07-31 라벨 제거)
CANDIDATE_COLOR = (150, 150, 150)  # 게이트 통과 후보 손 — 획득 전 상황 확인용
HAND_CONNECTIONS = (   # MediaPipe 21점 골격 — 손목(0)에서 손가락 5줄 + 손바닥 가로줄
    (0, 1), (1, 2), (2, 3), (3, 4),           # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),           # 검지
    (9, 10), (10, 11), (11, 12),              # 중지
    (13, 14), (14, 15), (15, 16),             # 약지
    (0, 17), (17, 18), (18, 19), (19, 20),    # 새끼
    (5, 9), (9, 13), (13, 17),                # 손바닥
)
SHAPE_TAG = {"fist": "(F)", "finger": "(1)", "open": "(5)"}   # F=주먹, 1=한 손가락,
                                               # 5=손 펼침. 불명은 무표시.
                                               # 2026-08-03 개편: 탐색=5, temp=1


def draw_user_hands(frame, hand_selector):
    """추론이 실제로 잡고 있는 것을 **골격 선**으로 그린다.

    ★2026-08-04 사용자 요청("추적 박스로 표현하지 말고 선으로"): 상자는 대략의
    영역만 알려줄 뿐, 모델이 어디를 어떻게 잡았는지는 못 보여준다. 선으로 그리면
    포즈가 어깨·팔꿈치·손목을 제대로 물었는지, 손 21점이 손가락 모양을 맞게
    따라가는지가 한눈에 보인다 — 현장 진단이 훨씬 빨라진다.
      · 몸(노랑): 목·어깨선·양팔(어깨→팔꿈치→손목) — 앵커로 잡은 그 사람
      · 손(초록): 추적 손 21점 골격 + 중심점·모양 태그
      · 회색 점: 게이트는 통과했지만 추적되지 않은 후보 손
    """
    for start_point, end_point in _body_lines(hand_selector):
        cv2.line(frame, (int(start_point[0]), int(start_point[1])),
                 (int(end_point[0]), int(end_point[1])), LOCK_COLOR, 2)
        for point in (start_point, end_point):
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, LOCK_COLOR, -1)
    if getattr(hand_selector, "anchor_head_box", None) is not None:
        x1, y1, x2, y2 = hand_selector.anchor_head_box
        # 머리는 상자 대신 원 — 크기(귀-귀 폭)는 도달 반경의 자라 계속 보여야 한다
        radius_px = max(2, (x2 - x1) // 2)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius_px, LOCK_COLOR, 2)
    landmarks = _tracked_landmarks(hand_selector)
    if landmarks is not None:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            cv2.line(frame,
                     (int(landmarks[start_idx][0]), int(landmarks[start_idx][1])),
                     (int(landmarks[end_idx][0]), int(landmarks[end_idx][1])),
                     TRACKED_COLOR, 2)
    for point in hand_selector.candidate_points():
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, CANDIDATE_COLOR, 1)
    signal = hand_selector.user_hand_signal()
    if signal is not None:
        shape, point = signal[0], signal[1]   # (모양, 좌표, 라벨, 검지비율)
        x_px, y_px = int(point[0]), int(point[1])
        cv2.circle(frame, (x_px, y_px), 10, TRACKED_COLOR, 2)
        cv2.putText(
            frame, "H" + SHAPE_TAG.get(shape, ""), (x_px + 12, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACKED_COLOR, 2,
        )
    return frame


def _body_lines(hand_selector):
    """앵커 사람의 골격 선 — 셀렉터가 못 주면 빈 목록(구 버전 호환)."""
    getter = getattr(hand_selector, "body_lines", None)
    return getter() if callable(getter) else []


def _tracked_landmarks(hand_selector):
    """추적 손 21점 — 셀렉터가 못 주면 None(구 버전 호환)."""
    getter = getattr(hand_selector, "tracked_hand_landmarks", None)
    return getter() if callable(getter) else None



def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-16).

    SCALE=어깨 스케일 / ARM=활성 팔+손 모양(원시 판별) / RET=복귀 삼킴 예약 방향 /
    SWIPE=진행도(±1.0 판정) / LATCH=고정 모양(F=주먹, 1=한 손가락, 5=손 펼침, -=없음)과
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
            # cv2.putText는 한글을 못 그린다(2026-08-03 실기 사진: "(대기)"가 ?????로
            # 찍혔다) — 계기판 문자열은 ASCII만 쓴다
            if low is not None else f"TAP idx {index_ratio:.2f} (waiting)")
    grab_repeat = debug.get("grab_repeat")
    if grab_repeat:
        # 폈다 쥘 때마다 오른다 — 안 오르면 래치 전환 실패, 0으로 돌아가면 창 초과·이동
        lines.append(f"GRAB {debug.get('grab_count', 0)}/{grab_repeat}")
    if "bend_posture" in debug:
        # 꺾임 판(hand_bend) 계기 — deg=손 방향 화면 각도(?=판별 불가), 자세 이름,
        # hold=유지 진행도(1.00 도달 = 발화 시점).
        # F=주먹 외적 부호(8차부터 계측 전용 — 뒤집기 폐기),
        # ok=엄지·검지 고리 닫힘 비율(임계 ok_touch_palm_ratio 미만이어야 OK —
        # confirm이 안 잡히면 이 값이 임계 아래로 내려가는지부터 볼 것)
        bend_deg = debug.get("bend_deg")
        deg_tag = "?" if bend_deg is None else f"{bend_deg:+.0f}"
        bend_z = debug.get("bend_z")
        z_tag = "" if bend_z is None else f" z{bend_z:+.2f}"
        fist_sign = debug.get("bend_fist_sign")
        fist_tag = "" if fist_sign is None else f" F{fist_sign:+.2f}"
        ok_ratio = debug.get("bend_ok_ratio")
        ok_tag = "" if ok_ratio is None else f" ok{ok_ratio:.2f}"
        lines.append(f"BEND {deg_tag}deg {debug.get('bend_posture') or '-'}"
                     f" hold {debug.get('bend_hold_ratio', 0):.2f}"
                     f"{z_tag}{fist_tag}{ok_tag}")
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
