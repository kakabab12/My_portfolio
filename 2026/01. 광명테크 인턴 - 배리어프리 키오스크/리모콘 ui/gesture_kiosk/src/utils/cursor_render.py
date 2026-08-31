"""헤드트래커 커서 그리기 — 세 트래커(head·eyebrow·forehead) 공용 (2026-08-31 신설).

왜 따로 뺐나
------------
세 파일에 **완전히 같은 draw_cursor가 각각 복사돼 있었다**(2026-08-31 확인:
세 함수의 해시가 동일). 커서 모양을 고치려면 세 곳을 똑같이 고쳐야 했고,
한 곳만 고치면 트래커마다 커서가 달라 보이게 된다. 한 곳으로 모은다.

무엇을 고쳤나
-------------
1) **안티에일리어싱** — 셋 다 cv2.LINE_AA 없이 그리고 있어서 원과 십자에
   계단이 보였다. 커서는 화면에서 가장 오래 보는 물체라 여기서 티가 난다.

2) **부분 픽셀 위치** — 예전엔 int(비율 x 폭)으로 잘라서, 커서를 아주 천천히
   움직이면 1픽셀씩 **툭툭 건너뛰었다**. 헤드트래커는 원래 미세 조정이 어려운
   입력이라 이 끊김이 그대로 "정밀도가 없다"는 느낌으로 이어진다.
   OpenCV의 shift 인자로 1/8픽셀 단위로 그려 움직임이 이어져 보이게 한다.

3) **대비 테두리(halo)** — 초록 커서 하나만 그리면 밝은 배경이나 초록 계열
   화면에서 사라진다. 배리어프리 키오스크는 어떤 화면 위에 올라갈지 모른다.
   커서 바깥과 안쪽에 어두운 테두리를 함께 그려 **어떤 배경에서도** 보이게 한다.
   (WCAG 2.2의 비텍스트 대비 원칙 — 조작에 필요한 그래픽은 인접 색과 3:1 이상.
    테두리를 두면 배경이 무엇이든 커서 경계에서 그 대비가 성립한다.)

4) **가운데 점** — 십자만 있으면 "정확히 어디를 가리키는가"가 선의 교차점이라
   눈으로 찍기 어렵다. 작은 점을 채워 조준점을 분명히 한다.

★안티에일리어싱과 투명색의 충돌 (2026-08-31 실기 보고로 발견)
--------------------------------------------------------------
오버레이 창은 마젠타(255,0,255)를 투명색으로 지정해 띄운다
(SetLayeredWindowAttributes + LWA_COLORKEY). 이 방식은 **그 색과 정확히
일치하는 픽셀만** 투명하게 만든다.

그런데 안티에일리어싱은 가장자리에서 커서 색과 배경색을 섞은 중간색을
만든다. 배경이 마젠타면 그 중간색은 마젠타가 아니므로 투명 처리가 안 되고,
**커서 둘레에 분홍 테두리로 보인다** (사용자 실기 보고: "커서 뒤에 분홍색").

그래서 **바깥을 향한 요소는 안티에일리어싱을 쓰지 않는다**:
  · 대비 테두리(halo), 진행 링, 드래그 채움 -> 계단이 지더라도 경계가 딱
    떨어져야 한다. 어차피 halo는 얇고 검어서 계단이 잘 안 보인다.
  · 본체 원·십자·조준점 -> 이미 halo가 깔린 **위에** 그려지므로, 섞이는
    상대가 마젠타가 아니라 검정이다. 여기는 안티에일리어싱을 그대로 쓴다.
즉 눈에 띄는 곡선은 여전히 매끄럽고, 투명색과 닿는 가장자리만 딱 떨어진다.

비용
----
전부 커서 한 개 크기(약 70x70) 안에서만 일어난다. 화면 전체를 다시 그리는
비용에 비하면 무시할 수준이고, 실제로 지우는 범위도 cursor_reach_px()로
그만큼만 계산한다(더티 사각형 — 각 트래커의 _blank_rect 설명 참고).
"""
import cv2

# 커서 크기 — 화면 해상도 캔버스 기준(카메라 프레임이 아니라 전체 화면에
# 그리므로 크게 잡는다). 세 트래커가 같은 값을 쓰도록 여기서 한 번만 정의한다
CURSOR_RADIUS_PX = 28
CURSOR_MARKER_SIZE_PX = 22
CURSOR_THICKNESS_PX = 3

# 조준점 — 십자 교차점을 눈으로 찍기 쉽게
CURSOR_DOT_RADIUS_PX = 3

# 대비 테두리 — 배경이 무엇이든 커서가 보이게 한다.
# 검정을 쓰는 이유: 키오스크 화면은 대체로 밝고, 어두운 화면에서는 커서
# 본체(밝은 초록)가 이미 잘 보인다. 즉 둘이 서로의 약점을 덮는다
CURSOR_HALO_COLOR = (0, 0, 0)
CURSOR_HALO_THICKNESS_PX = 2

# 부분 픽셀 정밀도 — OpenCV의 shift 인자는 좌표를 1/(2**shift) 픽셀 단위로
# 해석한다. 3이면 1/8픽셀이라 육안으로는 완전히 이어져 보이고, 좌표가
# 정수 범위를 넘칠 걱정도 없다(1/8픽셀 단위로도 화면 좌표는 충분히 작다)
_SUBPIXEL_SHIFT = 3
_SUBPIXEL_SCALE = 1 << _SUBPIXEL_SHIFT


def _pt(x_px, y_px):
    """부분 픽셀 좌표 -> OpenCV shift 정수 좌표."""
    return (int(round(x_px * _SUBPIXEL_SCALE)), int(round(y_px * _SUBPIXEL_SCALE)))


def _len(value_px):
    return int(round(value_px * _SUBPIXEL_SCALE))


def cursor_reach_px(progress_ring=True):
    """커서가 실제로 칠하는 최대 반경 — 더티 사각형 계산용.

    이 값이 실제보다 작으면 지우다 만 자국(잔상)이 남는다. 그리는 요소 중
    가장 바깥(진행 링 또는 원 + 테두리)에 선 두께와 안티에일리어싱 번짐
    여유를 더한다.
    """
    outer = CURSOR_RADIUS_PX + (8 if progress_ring else 0)
    return outer + CURSOR_THICKNESS_PX + CURSOR_HALO_THICKNESS_PX + 2


def draw_cursor(frame, cursor_x_ratio, cursor_y_ratio, recenter_progress_ratio=0.0,
                color=(0, 220, 0), filled=False, progress_color=(0, 165, 255)):
    """커서를 화면 비율 좌표(0~1)에 그린다.

    color/filled는 클릭·드래그 피드백용. 드래그 중엔 속을 채워서, 색 구분이
    어려운 사람도 "지금 누르고 있다"를 **형태로** 알 수 있게 한다.

    recenter_progress_ratio(0~1)가 0보다 크면 둘레에 진행 링을 함께 그린다 —
    재정렬까지 몇 초 남았는지 숫자 없이 보이게 한다. 진행 표시를 커서 자리에
    두는 이유: 다른 곳에 그리면 확인하려 시선을 옮기는 순간 커서가 반경을
    벗어나 응시가 끊긴다.
    """
    if cursor_x_ratio is None or cursor_y_ratio is None:
        return frame

    h_px, w_px = frame.shape[:2]
    x_px = cursor_x_ratio * w_px
    y_px = cursor_y_ratio * h_px
    center = _pt(x_px, y_px)

    # 1) 바깥 대비 테두리 — 본체보다 먼저 그려 본체가 위에 얹히게 한다
    cv2.circle(frame, center, _len(CURSOR_RADIUS_PX),
               CURSOR_HALO_COLOR, CURSOR_THICKNESS_PX + CURSOR_HALO_THICKNESS_PX * 2,
               cv2.LINE_8, _SUBPIXEL_SHIFT)      # 투명색과 닿는 가장자리 - 위 설명 참고

    # 2) 드래그 중이면 속을 채운다 (형태로도 상태를 알린다)
    if filled:
        cv2.circle(frame, center, _len(CURSOR_RADIUS_PX - 6), color, -1,
                   cv2.LINE_8, _SUBPIXEL_SHIFT)   # 원 안쪽은 아직 투명색이다

    # 3) 본체 원
    cv2.circle(frame, center, _len(CURSOR_RADIUS_PX), color, CURSOR_THICKNESS_PX,
               cv2.LINE_AA, _SUBPIXEL_SHIFT)

    # 4) 십자 — drawMarker는 부분 픽셀을 못 받으므로 선 두 개로 직접 그린다.
    #    테두리를 먼저 굵게 깔고 본체를 얹어 배경과 관계없이 보이게 한다
    half = CURSOR_MARKER_SIZE_PX / 2.0
    arms = (((x_px - half, y_px), (x_px + half, y_px)),
            ((x_px, y_px - half), (x_px, y_px + half)))
    for (ax, ay), (bx, by) in arms:
        cv2.line(frame, _pt(ax, ay), _pt(bx, by), CURSOR_HALO_COLOR,
                 CURSOR_THICKNESS_PX + CURSOR_HALO_THICKNESS_PX * 2,
                 cv2.LINE_8, _SUBPIXEL_SHIFT)     # 투명색과 닿는 가장자리
    for (ax, ay), (bx, by) in arms:
        cv2.line(frame, _pt(ax, ay), _pt(bx, by), color, CURSOR_THICKNESS_PX,
                 cv2.LINE_AA, _SUBPIXEL_SHIFT)

    # 5) 조준점 — 정확히 어디를 가리키는지
    if not filled:
        cv2.circle(frame, center, _len(CURSOR_DOT_RADIUS_PX), color, -1,
                   cv2.LINE_AA, _SUBPIXEL_SHIFT)

    # 6) 재정렬 진행 링
    if recenter_progress_ratio > 0.0:
        end_angle_deg = 360.0 * recenter_progress_ratio
        radius = _len(CURSOR_RADIUS_PX + 8)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle_deg,
                    progress_color, CURSOR_THICKNESS_PX, cv2.LINE_8, _SUBPIXEL_SHIFT)
    return frame
