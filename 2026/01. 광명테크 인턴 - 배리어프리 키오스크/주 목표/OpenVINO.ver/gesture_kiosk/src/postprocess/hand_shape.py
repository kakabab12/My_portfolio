"""postprocess 모듈 — 손 모양 판별: 주먹 / 한 손가락 (2026-07-23 새 스펙).

「제스처 정의 보고서」(2026-07-22 회사 확정)의 손 모양 기준 체계를 구현한다:
- **한 손가락** + 상·하·좌·우 이동 = 포커스 이동 (탐색 계층 — 화면 안 바뀜)
- **주먹** + 위/왼쪽/오른쪽 = 처음으로/이전/확인 (명령 계층 — 화면 바뀜)

wholebody 133 키포인트의 손 21점 **기하 규칙만** 쓴다 — 별도 손 모양 CNN 없음
(새 모델·새 의존성 0 = 상업 사용·코드 공개·저작권 공개 의무 없음,
rtmlib Apache-2.0 기존 검토 그대로).

판별 규칙 v2 (2026-07-23 2차 — 원근 단축 오판 대응, 실기 사진 2장 실증):
손가락(검지~새끼)을 3단계로 판정한다 —
- **폄**: 손끝-손목뿌리 거리가 둘째 관절(PIP)-뿌리 거리의 extend_ratio배 이상
- **굽힘(확인된 것만)**: ①관절 진행 방향이 반전(뿌리→PIP 방향 vs DIP→손끝 방향이
  반대 — 되접힌 손가락의 고유 기하: 펴진 손가락은 어느 각도로 찍어도 직선 투영) 또는
  ②손끝이 PIP보다 안쪽(curl_confirm_ratio 이하)
- **판단 불가(기권)**: 짧지만 접힘이 확인 안 됨 — 손가락이 카메라 쪽으로 누워
  투영이 짧아진 것(원근 단축)일 수 있다. v1은 이걸 굽힘으로 세어 **화면을 가리키는
  한 손가락이 주먹으로 오판**됐다(실기: right가 ok로 둔갑 — 가장 비싼 오발)

손 판정: 한 손가락 = 폄 1개 + 굽힘 확인 2개 이상 · 주먹 = 폄 0 + **기권 0** +
굽힘 확인 min_valid_fingers개 이상(기권이 하나라도 있으면 주먹을 단정하지 않는다)
· 그 외 = None(모양 불명 — 이동 추적은 하되, 확정은 다수결·모양 기억이 담당).
엄지는 세지 않는다 — 주먹을 쥐어도 엄지는 밖으로 삐져나와 오판이 잦고,
보고서의 "손가락 종류 무관"과도 합치한다 (검지~새끼 중 아무거나 1개).
"""
import math

# COCO-WholeBody 133 규격의 손 키포인트 (몸 0~16은 COCO 17과 동일).
# 손 21점 = 손목뿌리 1 + 손가락 4점(MCP·PIP·DIP·TIP)×5. 모델 좌표계(화면 기준)
# 좌/우이며, 거울 보정은 person_lock.user_side_points가 담당한다.
HAND_LAYOUT = {
    "left": {
        "root": 91,
        # 검지·중지·약지·새끼 — (MCP, PIP, DIP, TIP). 엄지(92~95)는 제외 (모듈 주석)
        "fingers": ((96, 97, 98, 99), (100, 101, 102, 103),
                    (104, 105, 106, 107), (108, 109, 110, 111)),
        "all": range(91, 112),
    },
    "right": {
        "root": 112,
        "fingers": ((117, 118, 119, 120), (121, 122, 123, 124),
                    (125, 126, 127, 128), (129, 130, 131, 132)),
        "all": range(112, 133),
    },
}
WHOLEBODY_KPT_COUNT = 133

SHAPE_FIST = "fist"       # 주먹 — 명령 계층
SHAPE_FINGER = "finger"   # 한 손가락 — 탐색 계층


def _confident_point(keypoints, index, min_conf):
    x, y, conf = keypoints[index]
    if conf < min_conf:
        return None
    return float(x), float(y)


def classify_hand_shape(keypoints, model_side, min_conf, extend_ratio,
                        min_valid_fingers, curl_confirm_ratio, fold_cos_margin=-0.3):
    """손 모양 판별 v2 -> "fist" | "finger" | None (모양 불명).

    keypoints: PersonPose.keypoints (wholebody 133 필요 — body 17이면 None).
    model_side: 모델 좌표계 기준 "left"/"right".
    손가락별 3단계(폄/굽힘 확인/기권 — 모듈 주석)로 세고,
    주먹은 기권이 하나라도 있으면 단정하지 않는다(원근 단축 오판 차단).
    fold_cos_margin: 되접힘(굽힘) 판정의 코사인 여유(2026-07-24, 오탐 신고 대응) —
    0.0(직각)이 아니라 이만큼 더 꺾여야 굽힘 확정으로 센다. 직각 근처의 키포인트
    노이즈로 부호가 흔들려 굽힘 오판정(→주먹/한 손가락 오탐)이 나는 걸 막는다.
    음수가 클수록 보수적 — 애매하면 기권(uncertain)이 늘어 단정을 줄인다.
    """
    if len(keypoints) < WHOLEBODY_KPT_COUNT:
        return None
    layout = HAND_LAYOUT[model_side]
    root = _confident_point(keypoints, layout["root"], min_conf)
    if root is None:
        return None

    extended_count = 0
    curled_count = 0
    uncertain_count = 0
    for finger_idx, (mcp, pip, dip, tip) in enumerate(layout["fingers"]):
        pip_point = _confident_point(keypoints, pip, min_conf)
        tip_point = _confident_point(keypoints, tip, min_conf)
        if pip_point is None or tip_point is None:
            continue   # 미관측 — 판정에 넣지 않는다 (기권과 구분: 짧다는 증거도 없음)
        pip_dist = math.dist(pip_point, root)
        if pip_dist <= 0.0:
            continue
        ratio = math.dist(tip_point, root) / pip_dist
        # 방향(코사인)을 폄/굽힘 판정보다 먼저 구한다 — 되접힘(방향 반전)이 둘 다에
        # 필요한 공용 신호다(2026-07-24 실측: 원근 단축에 비율보다 불변이도록 설계).
        # 펴진 채 카메라 쪽으로 누운 손가락은 짧아도 방향이 바깥으로 유지된다.
        mcp_point = _confident_point(keypoints, mcp, min_conf)
        dip_point = _confident_point(keypoints, dip, min_conf)
        is_folded = False
        cos_angle = None
        if mcp_point is not None and dip_point is not None:
            base_dx = pip_point[0] - mcp_point[0]
            base_dy = pip_point[1] - mcp_point[1]
            tip_dx = tip_point[0] - dip_point[0]
            tip_dy = tip_point[1] - dip_point[1]
            base_len = math.hypot(base_dx, base_dy)
            tip_len = math.hypot(tip_dx, tip_dy)
            if base_len > 0.0 and tip_len > 0.0:
                cos_angle = (base_dx * tip_dx + base_dy * tip_dy) / (base_len * tip_len)
                is_folded = cos_angle < fold_cos_margin
        # 2026-07-27 버그 수정 2차(실기 진단 로그 — 손가락 하나 폈는데 주먹으로 오분류
        # 재신고): 비율만으로 "폄"을 정하면, 카메라 정면에 가까운 각도로 가리키는
        # 손가락은 화면상 짧게 찍혀 extend_ratio를 못 넘고 계속 "기권"에 갇힌다(실측:
        # 검지 ratio가 1.17~1.47 사이를 맴돌며 1.5(구값)를 못 넘김 — 방향은 cos=0.9~1.0로
        # 명백히 안 굽힘). 그렇다고 ratio 문턱만 낮추면, 어중간한 자세에서 "굽힘 확정"인데
        # ratio만 우연히 높은 손가락(실측: ratio=1.21,cos=-0.91)까지 폄으로 잘못 셀 위험이
        # 있다 — 그래서 "폄"도 비율 AND 방향(안 굽힘) 둘 다 요구하도록 방어선을 추가했다.
        # is_folded가 이미 위에서 방향 반전을 걸러주므로, 여기 통과하면 카메라를 거의
        # 정면으로 가리키는 특수 케이스(point_camera류: 방향은 유지되지만 짧음)만
        # extend_ratio가 여전히 걸러 기권으로 남는다.
        if ratio >= extend_ratio and not is_folded:
            extended_count += 1
            continue
        # 2026-07-27 버그 수정(진단 로그로 실측 확인 — 오탐 신고 재현): 기존엔 "is_folded(방향)
        # or ratio<=curl_confirm_ratio"로 OR 결합돼 있어, 방향이 명백히 "안 굽힘"(cos 양수)이어도
        # 원근 단축으로 비율만 낮으면 그냥 굽힘 처리됐다 — 방향 판정이 비율에 의해 무력화되는
        # 구조적 오류(광각 카메라일수록 원근 단축이 커서 자주 발생, 실측 로그로 cos=0.5~0.99인데
        # 굽힘 판정된 사례 다수 확인). 방향 정보(cos_angle)가 있으면 **방향만** 믿는다(원근 단축에
        # 불변이도록 설계된 지표이므로) — 비율은 관절 미관측으로 방향을 못 구할 때의 대체 수단으로만 쓴다
        if cos_angle is not None:
            is_curled = is_folded
        else:
            is_curled = ratio <= curl_confirm_ratio
        if is_curled:
            curled_count += 1
        else:
            uncertain_count += 1   # 원근 단축 의심 — 기권

    if extended_count == 1 and curled_count >= min_valid_fingers - 1:
        return SHAPE_FINGER
    if extended_count == 0 and uncertain_count == 0 and curled_count >= min_valid_fingers:
        return SHAPE_FIST
    return None   # 기권 포함·2개 이상 폄·관측 부족 — 단정하지 않는다


def hand_center_point(keypoints, model_side, min_conf, min_points):
    """손 중심 추적점 — 신뢰도 통과 손 키포인트들의 평균 (x_px, y_px) | None.

    단일 점(손끝·손목) 대신 평균인 이유: 주먹↔한 손가락 어느 모양에서도 좌표가
    연속이고, 개별 점 오검출이 평균에 희석돼 궤적이 튀지 않는다. min_points
    미만이면 손이 사실상 안 보이는 것 — 추적하지 않는다 (폴백 없음, 2026-07-23
    스펙: 손목·팔꿈치 폴백 제거).
    """
    if len(keypoints) < WHOLEBODY_KPT_COUNT:
        return None
    points = []
    for index in HAND_LAYOUT[model_side]["all"]:
        point = _confident_point(keypoints, index, min_conf)
        if point is not None:
            points.append(point)
    if len(points) < min_points:
        return None
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )
