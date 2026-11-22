"""오른쪽 로봇 그림 (teleop.py · robot_server.py 가 같이 씀, 2026-11-15). mediapipe 필요 없음."""
import cv2
import numpy as np

from geometry import A4_D, A4_W
from piper_kin import TABLE_Z

DISP_H = 540
PANEL_W = 400


def robot_panel(kin, q, target, x0, y0, width, follow, rec, extra=()):
    """위에서 본 로봇 책상 (위) + 옆에서 본 모습 (아래). extra = 옆 그림 오른쪽에 쓸 짧은 글 (연결 상태 등)."""
    img = np.full((DISP_H, PANEL_W, 3), 245, np.uint8)
    pts = kin.link_points(q)
    s = 420.0                                                  # 1 m = 420 px
    # 위에서 본 그림: 로봇 앞(x) = 화면 위, 로봇 왼쪽(y) = 화면 왼쪽
    def top(p):
        return (int(PANEL_W / 2 - p[1] * s), int(255 - p[0] * s))
    corners = [(x0 - A4_D / 2, y0 + A4_W / 2), (x0 - A4_D / 2, y0 - A4_W / 2),
               (x0 + A4_D / 2, y0 - A4_W / 2), (x0 + A4_D / 2, y0 + A4_W / 2)]
    cv2.polylines(img, [np.array([top((a, b)) for a, b in corners])], True, (180, 180, 180), 1)
    cv2.drawMarker(img, top((x0, y0)), (120, 120, 120), cv2.MARKER_CROSS, 12, 1)
    cv2.circle(img, top((0, 0)), 14, (90, 90, 90), -1)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, top(a), top(b), (60, 60, 60), 3)
    if target is not None:
        cv2.circle(img, top(target), 6, (0, 0, 255), 2)
    cv2.circle(img, top(pts[-1]), 5, (0, 160, 0), -1)
    cv2.putText(img, "robot desk (top)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # 옆에서 본 그림: x = 오른쪽, z = 위
    y_tab = DISP_H - 18
    def side(p):
        return (int(30 + p[0] * s * 0.75), int(y_tab - (p[2] - TABLE_Z) * s * 0.75))
    cv2.line(img, (0, y_tab), (PANEL_W, y_tab), (150, 150, 150), 2)
    cv2.line(img, side((x0 - A4_D / 2, 0, TABLE_Z)), side((x0 + A4_D / 2, 0, TABLE_Z)), (180, 180, 180), 4)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, side(a), side(b), (60, 60, 60), 3)
    if target is not None:
        cv2.circle(img, side(target), 6, (0, 0, 255), 2)
    cv2.circle(img, side(pts[-1]), 5, (0, 160, 0), -1)
    cv2.putText(img, "robot (side)", (8, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    z = pts[-1][2] - TABLE_Z
    w_txt = f"gripper {width*1000:4.0f} mm" if np.isfinite(width) else "gripper ?"
    lines = [f"tip x {pts[-1][0]*100:5.1f} y {pts[-1][1]*100:5.1f} z {z*100:4.1f} cm",
             w_txt, "FOLLOW ON" if follow else "follow off (SPACE)"]
    if rec:
        lines.append("REC")
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 282 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 200) if t in ("REC", "FOLLOW ON") else (0, 0, 0), 1)
    for i, t in enumerate(extra):                              # 연결 상태: 옆 그림 오른쪽 빈 곳 ("!" 로 시작하면 빨강)
        cv2.putText(img, t.lstrip("!"), (212, 392 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200) if t.startswith("!") else (0, 0, 0), 1)
    return img
