"""postprocess 모듈 — 손 모양 판별: 주먹 / 한 손가락 (2026-07-23 새 스펙).

「제스처 정의 보고서」(2026-07-22 회사 확정)의 손 모양 기준 체계를 구현한다:
- **한 손가락** + 상·하·좌·우 이동 = 포커스 이동 (탐색 계층 — 화면 안 바뀜)
- **주먹** + 위/왼쪽/오른쪽 = 처음으로/이전/확인 (명령 계층 — 화면 바뀜)

판별 규칙 v3 (2026-07-28 — 입력을 wholebody 손 21점 → MediaPipe HandLandmarker
21점으로 교체, hand_tracker.py 참고):
- 좌표가 (x, y, z) 3차원이다 — v2가 2D 투영으로 구분 못 하던 원근 단축(카메라를
  가리키는 손가락)이 z 거리로 풀린다: 화면상 짧아도 3D로는 길다 → 폄.
- MediaPipe는 보이는 손만 보고하고 21점을 항상 채워 주므로 v2의 점별 신뢰도
  필터·"미관측" 처리가 필요 없다 (손 존재 신뢰도는 hand_tracker 옵션이 거른다).
- 3단계 판정(폄 / 굽힘 확인 / 기권)과 손 판정 규칙은 v2 그대로 유지 —
  실기에서 검증된 보수적 구조(기권이 있으면 주먹을 단정하지 않는다)를 계승.

손가락(검지~새끼) 3단계:
- **폄**: 손끝-손목뿌리 3D 거리가 둘째 관절(PIP)-뿌리 3D 거리의 extend_ratio배 이상
- **굽힘(확인된 것만)**: ①관절 진행 방향 반전(뿌리→PIP vs DIP→손끝 3D 내적 음수 —
  되접힌 손가락의 고유 기하) 또는 ②손끝이 PIP보다 안쪽(curl_confirm_ratio 이하)
- **판단 불가(기권)**: 짧지만 접힘이 확인 안 됨 — 단정하지 않는다

■ 미해결: 처진 손 오판 (2026-08-03 키오스크 실기 — 사용자 보고)
"팔이 가로로 되어 있고 손목이 아래로 쳐져 있으면 검지 주먹이 주먹으로 들어간다."
손을 옆에서 보면 MediaPipe가 가린 손가락을 추측으로 채워, 편 검지를 반쯤 접힌
점으로 찍는다. 랜드마크 자체가 틀리므로 아래 판별 규칙으로는 못 고친다.
**시도했다가 되돌린 것 2건** — 같은 길을 다시 파지 않도록 근거를 남긴다:

① PIP 관절각을 폄 인정 경로로 추가 (임계 101°)
   근거였던 실측: 검지 27°·90° / 주먹 137°·112° → 틈 11.5°
   결과: 실기 재확인에서 처진 검지 22%만 구제, 처진 주먹의 20%가 finger로 오판.
   자세가 조금만 달라지면 분포가 겹친다 — 한 세션의 좁은 틈이었다.

② 손바닥 정면도(법선의 카메라축 성분)로 "못 믿을 자세"를 감지해 판정 기권
   근거였던 실측: 좋은 자세 0.58~0.96 / 처진 손 0.00~0.28 → 틈 0.75
   결과: **정지 자세만 재서 생긴 착시**였다. 손을 움직이는 실사용에서는 정면도
   중앙값이 검지 0.283·주먹 0.205로 처진 손과 완전히 겹친다(임계 0.45에서
   정상 사용의 90%가 게이트에 걸렸다). 기권이 이어지자 래치가 직전 모양을
   붙들어 주먹·손바닥이 전부 finger로 나가는 더 큰 사고가 됐다.
   교훈: 판별 임계는 **움직이는 손**으로 재야 한다. 정지 측정은 실사용이 아니다.

뿌리 기준 비율로 자세별 전용 임계를 두는 것도 3라운드 반복 측정에서 겹쳤다
(편 검지 최저 0.618 vs 같은 손의 굽힌 손가락 최고 0.632 — 한 손 안에서 뒤집힌다).
다음에 손댈 사람에게: 이 자세는 **랜드마크 단계에서** 이미 정보가 뭉개진다.
판별 규칙을 고치는 방향이 아니라 관측 자체를 바꾸는 방향(카메라 각도·해상도,
또는 모양 판별을 손 실루엣 분류로 교체)으로 접근할 것.

손 판정: 한 손가락 = 폄 1개 + 굽힘 확인 min_valid_fingers-1개 이상 ·
주먹 = 폄 0 + **기권 0** + 굽힘 확인 min_valid_fingers개 이상 · 그 외 = None
(모양 불명 — 이동 추적은 하되, 확정은 다수결·모양 기억이 담당).
엄지는 세지 않는다 — 주먹을 쥐어도 엄지는 밖으로 삐져나와 오판이 잦고,
보고서의 "손가락 종류 무관"과도 합치한다 (검지~새끼 중 아무거나 1개).
"""
import math

# MediaPipe 21점 규격 (hand_tracker.HandDetection.landmarks 순서)
HAND_ROOT_IDX = 0
# 검지·중지·약지·새끼 — (MCP, PIP, DIP, TIP). 엄지(1~4)는 제외 (모듈 주석)
HAND_FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_KPT_COUNT = 21
THUMB_TIP_IDX = 4    # 엄지 손끝 — OK 사인(엄지·검지 고리) 판별에만 쓴다
INDEX_TIP_IDX = 8    # 검지 손끝
INDEX_MCP_IDX = 5    # 검지 뿌리 — 손바닥 폭(OK 고리 크기의 자)의 한쪽 끝
PINKY_MCP_IDX = 17   # 새끼 뿌리 — 손바닥 폭의 반대쪽 끝

SHAPE_FIST = "fist"       # 주먹 — 명령 계층
SHAPE_FINGER = "finger"   # 한 손가락 — 탐색 계층
SHAPE_OPEN = "open"       # 손바닥(전부 폄) — temp 계층 (2026-07-31 사용자 요청:
                          #   temp_left/temp_right/temp_top). 종전엔 "2개 이상 폄 =
                          #   불명"으로 버리던 모양을 셋째 모양으로 승격

STATE_EXTENDED = "extend"      # 폄
STATE_CURLED = "curl"          # 굽힘 확인
STATE_UNCERTAIN = "uncertain"  # 기권 — 접힘 증거 없음

def _dist3d(point_a, point_b):
    return math.dist(
        (float(point_a[0]), float(point_a[1]), float(point_a[2])),
        (float(point_b[0]), float(point_b[1]), float(point_b[2])),
    )


def finger_states(landmarks, extend_ratio, curl_confirm_ratio):
    """손가락(검지~새끼)별 (3D 비율, 판정 상태) 목록 — 판별 근거 계측(2026-07-28).

    classify_hand_shape가 이 계산을 그대로 쓰고, person_lock이 DEBUG 레벨에서
    이 값을 로그로 남긴다 — 실기에서 주먹/한 손가락의 비율 분포를 측정해
    extend_ratio·curl_confirm_ratio를 감이 아니라 데이터로 정하기 위한 계측이다.
    """
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return []
    root = landmarks[HAND_ROOT_IDX]
    states = []
    for mcp, pip, dip, tip in HAND_FINGERS:
        pip_dist = _dist3d(landmarks[pip], root)
        if pip_dist <= 0.0:
            continue
        ratio = _dist3d(landmarks[tip], root) / pip_dist
        if ratio >= extend_ratio:
            states.append((ratio, STATE_EXTENDED))
            continue
        # 짧다 — 진짜 굽힘인지 확인: 되접힘(방향 반전)이 굽힘의 고유 기하다
        base = landmarks[pip] - landmarks[mcp]
        tip_dir = landmarks[tip] - landmarks[dip]
        is_folded = float(
            base[0] * tip_dir[0] + base[1] * tip_dir[1] + base[2] * tip_dir[2]
        ) < 0.0
        if is_folded or ratio <= curl_confirm_ratio:
            states.append((ratio, STATE_CURLED))
        else:
            states.append((ratio, STATE_UNCERTAIN))
    return states


def classify_hand_shape(landmarks, extend_ratio, min_valid_fingers, curl_confirm_ratio):
    """손 모양 판별 v3 -> "fist" | "finger" | None (모양 불명).

    landmarks: HandDetection.**world_landmarks** — shape (21, 3), 미터 단위 월드
    좌표를 권장한다 (2026-07-28 실기 정정: 화면 좌표의 z는 노이즈가 커서 가리키기
    자세의 방향 반전 판정이 튀어 주먹 오판 재발 — 시점 불변 월드 기하로 판별).
    판별은 비율·방향만 쓰므로 스케일 무관 — 화면 좌표를 넣어도 동작은 한다.
    손가락별 3단계(폄/굽힘 확인/기권 — 모듈 주석)를 3D 거리로 세고,
    주먹은 기권이 하나라도 있으면 단정하지 않는다 (v2 보수 구조 유지).
    """
    states = finger_states(landmarks, extend_ratio, curl_confirm_ratio)
    if not states:
        return None
    extended_count = sum(1 for _, state in states if state == STATE_EXTENDED)
    curled_count = sum(1 for _, state in states if state == STATE_CURLED)
    uncertain_count = sum(1 for _, state in states if state == STATE_UNCERTAIN)

    if extended_count == 1 and curled_count >= min_valid_fingers - 1:
        return SHAPE_FINGER
    if extended_count == 0 and uncertain_count == 0 and curled_count >= min_valid_fingers:
        return SHAPE_FIST
    if extended_count >= min_valid_fingers and curled_count == 0:
        # 손바닥(2026-07-31) — 폄이 판정 정족수 이상 + 굽힘 확인 0: 블러로 한둘이
        # 기권이어도 나머지가 다 펴져 있으면 손바닥. 굽힘이 하나라도 확인되면
        # 불명(아래) — 한 손가락→손바닥 전환 중간 자세를 단정하지 않는다
        return SHAPE_OPEN
    return None   # 기권 혼재·폄 2개(중간 자세) — 단정하지 않는다


def ok_touch_ratio(landmarks):
    """엄지 끝↔검지 끝 3D 거리 / 손바닥 폭(5↔17) | None(퇴화) — 판별·계측 공용.

    OK 고리가 닫힐수록 작다(맞닿음 ~0.15-0.4, 펼친 손 ~0.6-1.3). 계기판
    (BEND 줄 ok=)이 이 값을 그대로 보여 실기에서 임계를 데이터로 맞춘다.
    """
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return None
    palm_span = _dist3d(landmarks[INDEX_MCP_IDX], landmarks[PINKY_MCP_IDX])
    if palm_span <= 0.0:
        return None
    return _dist3d(landmarks[THUMB_TIP_IDX], landmarks[INDEX_TIP_IDX]) / palm_span


def ok_touch_ratio_2d(landmarks_px):
    """엄지 끝↔검지 끝 **화면(2D)** 거리 / 손바닥 폭(5↔17 2D) | None.

    ★2026-08-06 손바닥쪽 OK 실기 실패의 진단 계측으로 신설 → 같은 날 판별
    (2D 구제 경로 — hand_bend._is_ok_posture)에 채용(사용자 제안 "엄지와
    검지가 맞닿았는지 로직만 추가"): 월드(3D) 고리 거리는 손바닥이 카메라를
    볼 때 검지 환각으로 부풀지만(실기 로그 ok=0.57~0.91 — 진짜 펼친 손
    0.58~1.3과 완전 겹침), 화면(2D) 랜드마크는 이미지에서 직접 회귀돼
    카메라 창 골격은 고리를 제대로 그린다 — 그 관측을 판별에 쓴다.
    """
    if landmarks_px is None or len(landmarks_px) < HAND_KPT_COUNT:
        return None
    palm_span_px = math.dist(landmarks_px[INDEX_MCP_IDX][:2],
                             landmarks_px[PINKY_MCP_IDX][:2])
    if palm_span_px <= 0.0:
        return None
    return math.dist(landmarks_px[THUMB_TIP_IDX][:2],
                     landmarks_px[INDEX_TIP_IDX][:2]) / palm_span_px


def is_ok_sign(landmarks, extend_ratio, curl_confirm_ratio, touch_palm_ratio):
    """OK 사인(엄지·검지 끝 맞닿은 고리 + 중지~새끼 폄) -> bool.

    2026-08-06 8차 신설(사용자 지정 — "ok 손표시 = confirm"). 판별 근거 둘:
    ① 엄지 끝(4)↔검지 끝(8)의 3D 거리가 손바닥 폭(검지 뿌리 5↔새끼 뿌리 17)의
       touch_palm_ratio 이하 — 고리가 닫혀 있다. 손바닥 폭 비례라 손 크기·
       카메라 거리 무관.
    ② 중지·약지·새끼 셋 다 "폄"(finger_states 3단계) — 펼친 배경 손가락이
       OK의 고유 실루엣이다. 이 조건이 주먹을 가른다: 주먹도 엄지가 검지에
       붙지만(①이 우연히 참일 수 있다) 나머지가 굽어 있다.
    검지 상태는 안 본다 — 고리 진 검지는 굽힘/기권/폄 어디로도 읽힐 수 있어
    (느슨한 고리는 폄으로 읽힌다) 상태를 조건으로 걸면 인식만 잃는다.
    ※classify_hand_shape보다 **먼저** 물어야 한다 — OK는 폄 3개·굽힘 확인 0으로
    SHAPE_OPEN(펼친 손)으로 오독되기 때문이다 (hand_bend._classify_posture).
    """
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return False
    states = finger_states(landmarks, extend_ratio, curl_confirm_ratio)
    if len(states) != len(HAND_FINGERS):
        return False
    if any(state != STATE_EXTENDED for _, state in states[1:]):
        return False   # 중지~새끼 중 안 편 손가락 — OK 실루엣이 아니다
    touch = ok_touch_ratio(landmarks)
    return touch is not None and touch <= touch_palm_ratio


def hand_center_point(landmarks):
    """손 중심 추적점 — 21점 화면 좌표의 평균 (x_px, y_px) | None.

    단일 점(손끝·손목) 대신 평균인 이유: 주먹↔한 손가락 어느 모양에서도 좌표가
    연속이고, 개별 점 흔들림이 평균에 희석돼 궤적이 튀지 않는다 (v2와 동일).
    z는 궤적(화면 이동) 판정에 쓰지 않으므로 제외한다.
    """
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return None
    return (
        float(sum(point[0] for point in landmarks) / len(landmarks)),
        float(sum(point[1] for point in landmarks) / len(landmarks)),
    )
