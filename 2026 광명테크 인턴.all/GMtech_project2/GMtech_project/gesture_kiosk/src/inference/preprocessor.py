"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

현재 백엔드(ultralytics)는 리사이즈·정규화를 내부에서 처리하므로
여기서는 거울 반전만 담당한다. 추후 TensorRT 바인딩을 직접 다루게 되면
letterbox·정규화·CHW 변환이 이 모듈로 들어온다.
"""
import math

import cv2
import numpy as np

BLUR_KSIZE = 13   # 가우시안 블러 커널(홀수 필수) — **축소된 프레임에** 적용된다.
                  # 원본 기준으로는 13 × BLUR_DOWNSCALE = 52 상당 (구 51과 동등)
BRIDGE_PX = 9     # 연결성 판정 전에 메울 틈의 크기(절반 해상도 기준 —
                  # 원본으로는 약 18px). 손목처럼 얇은 부위에서 마스크가 한두
                  # 화소 끊겨 **내 손이 별개 덩어리로 버려지는 것**을 막는다.
                  # 키우면 내 손이 안전해지지만 뒷사람과도 붙기 쉬워진다
BLUR_DOWNSCALE = 4  # 블러 전 축소 배율 — ★2026-08-07 성능(코드 리뷰 지적).
                  # 종전엔 720p 원본에 51×51을 매 프레임 돌렸다: 분리 필터라도
                  # 픽셀당 ~100연산 × 2.7M픽셀이라 이미 낮은 FPS(3.7~7.0)를 더
                  # 갉는다. 1/4로 줄여 블러하고 되돌리면 **연산량이 ~1/16**이 되고,
                  # 목적이 "디테일 파괴"라 축소-확대로 생기는 화질 차이는 오히려
                  # 취지에 부합한다(축소 자체가 저역통과다). 표준 기법


def _blur_whole(frame):
    """프레임 전체를 흐린 사본 — 축소→블러→확대로 원본 직접 블러를 대체한다.

    2026-08-07 신설. 두 블러 함수가 같은 비용 문제를 공유하므로 한 곳에 모았다
    (BLUR_DOWNSCALE 주석에 근거).
    """
    height_px, width_px = frame.shape[:2]
    small = cv2.resize(frame, (max(1, width_px // BLUR_DOWNSCALE),
                               max(1, height_px // BLUR_DOWNSCALE)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return cv2.resize(small, (width_px, height_px), interpolation=cv2.INTER_LINEAR)


def anchor_keep_sharp_box(anchor_frame, ratio_shoulder_widths, frame_width_px,
                          frame_height_px):
    """앵커 중심(x,y,어깨너비px) -> 선명하게 남길 사각형(x1,y1,x2,y2) | None.

    2026-08-07 신설(사용자 제안 — "인식되는 사람 제외하고 모든 것을 블러 처리"):
    앵커 사람 몸통 주변만 정사각형으로 남기고 프레임 경계로 잘라낸다. 반경은
    팔 도달(anchor_reach_px, roi_zoom이 쓰는 값)과 **독립**이다 — 그 반경은
    제스처를 놓치지 않으려고 넉넉히(성인 팔 도달) 잡혀 있어, 바로 옆에 앉은
    사람까지 같이 덮기 쉽다. 여기는 그보다 좁게 잡아 인접한 사람을 배제하는
    것이 목적이라 별도 설정(keep_sharp_shoulder_widths)으로 튜닝한다.
    """
    if anchor_frame is None:
        return None
    center_x_px, center_y_px, shoulder_width_px = anchor_frame
    if shoulder_width_px <= 0.0:
        return None
    half_px = ratio_shoulder_widths * shoulder_width_px / 2.0
    x1 = int(max(0.0, center_x_px - half_px))
    y1 = int(max(0.0, center_y_px - half_px))
    x2 = int(min(float(frame_width_px), center_x_px + half_px))
    y2 = int(min(float(frame_height_px), center_y_px + half_px))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def blur_outside_region(frame, region_box):
    """region_box(x1,y1,x2,y2) 안만 남기고 나머지는 강하게 흐린다.

    2026-08-07 신설 — 인식 대상(앵커) 밖의 사람을 흐려서 손·포즈 모델이 아예
    못 보게 만드는 실험적 방어(사용자 제안). region_box가 None이면 원본
    그대로(기능 비활성 시 종전 동작). 프레임을 복사하므로 원본은 안 건드린다.

    ★2026-08-07 2차(사용자 결정 — "블러를 사람 누끼 딴거 제외하고"): 이
    사각형은 옆에 붙어 앉은 사람과 원리적으로 분리가 안 된다 — 사각형 안에
    옆 사람 몸이 같이 들어가면 그대로 선명하게 남는다. 실루엣 마스크가
    있으면 아래 blur_outside_mask가 대신 쓰인다(realtime_loop) — 이 함수는
    마스크가 없을 때(비활성·추출 실패)의 폴백으로 남긴다.
    """
    if region_box is None:
        return frame
    x1, y1, x2, y2 = region_box
    blurred = _blur_whole(frame)
    blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return blurred


def keep_component_at(binary, seed_point):
    """seed_point가 속한 **연결 덩어리 하나만** 남긴 이진 마스크 — 1인 보장의 핵심.

    ★2026-08-07 신설(사용자 요청 — "모든 기술력·논문 참고해서 뒷사람 인식 안되게").
    연결 성분 분석(connected component labeling)은 영상처리의 표준 기법이고,
    여기서는 **기하학적 보장**을 준다: 앵커 몸에서 픽셀이 이어지지 않는 영역은
    무슨 일이 있어도 선명하게 남지 않는다. 마스크가 오작동해 뒷사람까지
    덮더라도, 두 사람이 화면에서 떨어져 있으면 덩어리가 갈리므로 뒷사람 쪽은
    통째로 빠진다. 임계값 튜닝이 아니라 위상(연결성)에 기대는 방어라
    조명·체형·거리에 흔들리지 않는다.
    ※한계: 두 사람이 화면에서 **맞닿으면** 한 덩어리가 되어 못 가른다 —
    그 경우는 exclude_mask(상대 실루엣 빼기)와 깊이 관문이 받는다.

    seed_point가 없거나 마스크 밖이면 원본 그대로(방어) — 씨앗을 못 믿을 때
    엉뚱한 덩어리를 고르느니 종전 동작이 낫다.
    """
    if seed_point is None:
        return binary
    if binary.dtype != np.uint8:
        # cv2 연산은 bool 배열을 못 받는다 — 호출부마다 변환을 신경 쓰느니
        # 여기서 흡수한다(반환은 항상 uint8 0/1)
        binary = binary.astype(np.uint8)
    seed_x, seed_y = int(seed_point[0]), int(seed_point[1])
    height_px, width_px = binary.shape[:2]
    if not (0 <= seed_x < width_px and 0 <= seed_y < height_px):
        return binary
    if not binary[seed_y, seed_x]:
        return binary   # 앵커 중심이 실루엣 밖 — 관측이 어긋난 상태라 손대지 않는다
    # 연결성 판정만 절반 해상도에서 한다(실측 14.2ms -> 약 1/4). **연결성은
    # 저해상도에서도 보존된다** — 팔처럼 얇은 부위도 원본 40px면 20px라 안
    # 끊긴다(1/4로 더 줄이면 손목이 끊겨 사용자 손이 잘릴 위험이 있어 1/2까지만).
    # 최종적으로 원본 마스크와 AND 해서 **경계 정밀도는 원본 그대로** 유지한다
    small = cv2.resize(binary, (max(1, width_px // 2), max(1, height_px // 2)),
                       interpolation=cv2.INTER_NEAREST)
    # ★연결성을 따지기 전에 작은 틈을 메운다(2026-08-07 사용자 보고 —
    # "뒷사람이 있으면 손 관절값도 안 보인다"). 세그멘테이션은 손목처럼 얇은
    # 부위에서 몸과의 연결이 한두 화소 끊길 수 있는데, 그대로 판정하면 **내
    # 손이 별개 덩어리가 되어 통째로 버려진다** — 방어가 사용자를 지우는
    # 최악의 형태다. 미리 부풀려 붙여 놓고 판정하면 그 오검이 사라진다.
    # 최종적으로 원본과 AND 하므로 부풀린 만큼이 남지는 않는다
    bridged = cv2.dilate(small, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BRIDGE_PX, BRIDGE_PX)))
    count, labels = cv2.connectedComponents(bridged, connectivity=8)
    if count <= 2:
        return binary   # 덩어리가 하나뿐 — 걸러낼 것이 없다
    seed_label = labels[min(seed_y // 2, labels.shape[0] - 1),
                        min(seed_x // 2, labels.shape[1] - 1)]
    if seed_label == 0:
        return binary   # 축소 과정에서 씨앗이 배경으로 떨어졌다 — 손대지 않는다
    chosen = (labels == seed_label).astype(np.uint8)
    chosen = cv2.resize(chosen, (width_px, height_px), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(binary, chosen)


def arm_reach_mask(shape, body_points, radius_px):
    """앵커의 팔 주변만 1인 마스크 — **옆 사람 차단**의 기하학적 근거.

    ★2026-08-07 신설(사용자 보고 — "옆에 사람있으니까 블러 처리됐다가 손은
    보이던데"). 옆 사람은 뒷사람과 기하가 정반대라 깊이 관문이 원리적으로
    안 듣는다 — 옆에 서면 카메라와의 거리가 거의 같기 때문이다.
    사각형(bound_box)으로도 못 가른다: 내 팔 도달을 다 담으려면 한 변이
    어깨너비의 3.6배는 돼야 하는데, 그 안에 옆 사람이 통째로 들어온다.

    이 코드베이스는 같은 문제를 손 판정에서 이미 풀었다(2026-08-03 손목 반경
    도입 근거 — hand_select._filter_hands_by_anchor): **판정 영역이 내 팔을
    따라다니면** 팔을 내렸을 때 옆 사람 자리를 안 덮는다. 같은 원리를 블러에
    적용한다 — 앵커의 팔 관절(어깨·팔꿈치·손목) 주변만 선명하게 남긴다.
    옆 사람 손은 내 팔에서 멀어 흐려지고, 내 손은 내 손목 옆이라 살아남는다.

    ★블러의 유일한 소비자가 **손 모델**이라는 점이 이 좁힘을 가능하게 한다
    (포즈는 블러 안 된 프레임을 따로 본다 — realtime_loop). 몸통 전체를
    선명하게 남길 이유가 없다.

    body_points가 비었으면 None(제약 없음) — 관절을 못 믿을 때 내 손을
    자르느니 방어를 거르는 이 파일의 일관된 원칙.
    """
    if not body_points or radius_px <= 0:
        return None
    reach = np.zeros(shape[:2], dtype=np.uint8)
    for point in body_points:
        cv2.circle(reach, (int(point[0]), int(point[1])), int(radius_px), 1, -1)
    return reach


def hand_box_reach_mask(shape, hand_box, padding_ratio):
    """추적 중인 손의 **최근 위치** 주변만 1인 마스크 — 관절 반경의 사각지대 보강.

    ★2026-08-07 4차(사용자 보고 — "제스처를 해도 블러 때문에 초록색 관절값이
    사라진다"). arm_reach_mask는 어깨·팔꿈치·손목 같은 **몸 관절**에서 반경을
    잰다. 그런데 실측(같은 날 화면 캡처)으로 확인하니 손을 들거나 펼치는
    순간 손끝이 그 반경 경계에 바로 걸렸다 — 관절 반경이 손 자체의 위치가
    아니라 몸통 기준이라, 제스처로 손이 관절에서 조금만 더 뻗으면(손가락을
    펴거나 팔을 드는 정상적인 동작만으로도) 블러 쪽으로 넘어간다.

    해법은 관절이 아니라 **손이 실제로 있던 자리**를 기준으로 삼는 것이다.
    hand_selector.locked_box(추적 손 주변 EMA 박스, 이미 표시용으로 쓰던 값)의
    중심에 그 대각선 길이의 padding_ratio배 반경을 둔다 — 이전 프레임의 손
    위치이지만, 영상 프레임 간격에서 손이 그만큼 순간 이동하지는 않으므로
    좋은 예측치다. 손 모양이 크게 바뀌어도(주먹→편 손) 박스 자체가 그 변화를
    이미 담고 있어 반경이 따로 흔들리지 않는다.

    hand_box가 없으면(추적 중인 손이 없음) None — 이 경우 관절 반경만 적용된다.
    """
    if hand_box is None or padding_ratio <= 0:
        return None
    x1, y1, x2, y2 = hand_box
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    diagonal_px = math.hypot(x2 - x1, y2 - y1)
    radius_px = diagonal_px / 2.0 * padding_ratio
    return arm_reach_mask(shape, [center], radius_px)


def combine_reach_masks(*masks):
    """여러 반경 마스크를 하나로 합친다 — **어느 하나에만 들어도** 선명하게 남긴다.

    관절 반경(arm_reach_mask)과 손 위치 반경(hand_box_reach_mask)은 서로 다른
    사각지대를 메우는 관계라 합집합이어야 한다 — 교집합으로 하면 둘 다 통과한
    좁은 영역만 남아 오히려 더 많이 잘린다. None인 인자는 건너뛴다(그 반경이
    비활성이거나 기준점이 없다는 뜻 — 나머지 반경으로 판단한다).
    전부 None이면 None을 돌려줘 호출부가 "제약 없음"으로 처리하게 한다.
    """
    result = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else cv2.bitwise_or(result, mask)
    return result


def blur_outside_mask(frame, mask, mask_threshold, dilate_px=0, bound_box=None,
                      exclude_mask=None, seed_point=None, reach_mask=None):
    """실루엣 마스크(0~1, frame과 같은 h×w) 기준 임계 이상만 남기고 나머지는 흐린다.

    2026-08-07 신설(사용자 결정 — "블러를 사람 누끼 딴거 제외하고"):
    blur_outside_region의 사각형은 인접한 두 사람을 못 가르지만, Pose
    Landmarker가 사람별로 주는 픽셀 단위 실루엣 마스크를 쓰면 그 사람의
    윤곽 **밖**만 흐려져 바로 옆 사람도 실루엣 밖이면 블러된다.

    mask가 None이거나 frame과 크기가 다르면(마스크 미관측·해상도 불일치 등
    방어적 처리) 원본 그대로 — 블러 없음이 잘못된 블러보다 안전하다. 호출부
    (realtime_loop)가 이 경우 사각형 폴백(blur_outside_region)으로 넘어간다.

    ★2026-08-07 3차(사용자 실기 보고 — "손가락 손바닥 쪽에서 블러를 잘 못
    잡는다"): Pose 세그멘테이션은 몸통 실루엣 위주로 학습돼, 손가락처럼 얇고
    빠르게 움직이는(모션 블러 낀) 말단은 신뢰도가 몸통보다 낮게 나온다 —
    torso는 0.9~1.0인데 뻗은 손끝은 문턱(기본 0.5) 밑으로 떨어지기 쉽다.
    그러면 **내 손끝이 남의 것처럼 블러돼**, 정작 손 모델이 봐야 할 손 모양이
    입력 단계에서부터 뭉개진다 — 결과적으로 블러가 다른 사람이 아니라
    나 자신을 지워버리는 역효과. dilate_px(> 0)를 주면 문턱을 넘은 "선명
    영역"을 그만큼(픽셀) 팽창시켜 이 경계 손실을 흡수한다 — 신뢰도를 낮추는
    것(mask_threshold ↓)과 달리 배경의 애매한 픽셀을 끌어들이지 않고, 이미
    확실한 몸통 영역에서 바깥으로만 여유를 준다.
    """
    if mask is None or mask.shape[:2] != frame.shape[:2]:
        return frame
    blurred = _blur_whole(frame)
    # ★성능(2026-08-07 실측): 전 과정을 uint8 + OpenCV 연산으로 통일한다.
    # 종전엔 bool 배열의 fancy indexing(blurred[keep] = frame[keep])을 썼는데
    # 720p 2.7M 픽셀에서 그 한 줄만 30ms를 먹었다 — 전체 58ms의 절반 이상이다.
    # cv2 연산은 SIMD로 도는 C 구현이라 같은 일을 훨씬 싸게 한다
    keep = (mask >= mask_threshold).astype(np.uint8)
    # ★1인 보장은 **팽창·뺄셈보다 먼저**다(2026-08-07): 앵커 몸과 이어지지
    # 않는 덩어리를 여기서 떨어내야, 뒤이은 팽창이 그 덩어리를 키우지 않는다
    keep = keep_component_at(keep, seed_point)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        keep = cv2.dilate(keep, kernel)
    if exclude_mask is not None and exclude_mask.shape[:2] == frame.shape[:2]:
        # ★"앵커 안 잡힌 사람은 반드시 흐리게"(2026-08-07 사용자 요청). 포즈는
        # 사람별 실루엣을 다 주는데 종전엔 앵커 것만 쓰고 나머지를 버렸다 —
        # 추정할 필요가 없는 정보를 안 쓰고 있었다. 그 합집합을 여기서 빼면
        # 두 실루엣이 겹치는 애매한 구간이 **흐리는 쪽으로** 판정된다(사용자가
        # 원한 방향). 가까이 붙을수록 앵커 마스크의 유출이 커지는데, 유출이
        # 정확히 상대 실루엣 위에서 일어나므로 이 뺄셈이 그 지점을 정조준한다.
        # ★순서가 중요하다 — **팽창(dilate) 뒤에** 뺀다: 먼저 빼면 팽창이 도로
        # 상대 몸으로 번져 뺀 의미가 없어진다(팽창은 내 손끝을 살리는 장치이지
        # 남을 덮으라는 장치가 아니다)
        # 0/1 uint8끼리의 포화 뺄셈 = AND NOT (keep 1·other 1 -> 0)
        keep = cv2.subtract(keep, (exclude_mask >= mask_threshold).astype(np.uint8))
    if reach_mask is not None and reach_mask.shape[:2] == frame.shape[:2]:
        # 앵커의 팔에서 먼 곳은 남기지 않는다 — 옆 사람 손 차단(위 함수 주석).
        # 팽창 뒤에 적용해야 팽창이 팔 밖으로 번지는 것도 함께 잘린다
        keep = cv2.bitwise_and(keep, reach_mask)
    if bound_box is not None:
        # ★1인 제한(2026-08-07 사용자 지적 — "블러 사람 한명만 인식하게해줘
        # 리미트가 설정안되있는거같은데"): 마스크는 실측상 사람별로 잘 갈리지만
        # (합성 2인 검증: 마스크0 = 오른쪽 20.4%/왼쪽 1.5%, 마스크1 = 그 반대)
        # 상대편으로 ~7% 유출이 있고, 팽창(dilate_px)이 그 유출을 더 키운다.
        # 앵커 사람이 있을 수 없는 자리는 **무조건** 흐린다 — 마스크가 어떻게
        # 새든 선명 영역이 이 사각형을 못 넘는다. 정밀도는 마스크가, 범위
        # 한계는 이 상자가 담당하는 이중 구조
        # 전체 크기 배열을 새로 만들지 않고 **바깥 네 구역만 0으로** 지운다 —
        # 종전 np.ones + 불리언 인덱싱은 같은 일에 배열 두 개를 더 만들었다
        x1, y1, x2, y2 = bound_box
        keep[:y1] = 0
        keep[y2:] = 0
        keep[:, :x1] = 0
        keep[:, x2:] = 0
    # cv2.copyTo는 마스크가 0이 아닌 화소만 복사한다 — 종전 fancy indexing
    # (blurred[keep] = frame[keep])보다 훨씬 싸다(위 성능 주석 참고)
    cv2.copyTo(frame, keep, blurred)
    return blurred


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]

    def preprocess_frame(self, frame):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전한다."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        return frame
