"""화면의 **실제 물리 크기**(mm)를 운영체제에서 알아낸다 (2026-09-05 신설).

왜 필요한가
-----------
커서가 "얼굴이 향한 곳"에 있으려면 화면이 실제로 얼마나 큰지 알아야 한다.
사용자가 화면에서 Z만큼 떨어져 고개를 θ 돌리면 시선이 화면에서 Z·tan(θ)만큼
옮겨가므로, 화면 절반 폭 W/2에 닿는 각도는 atan((W/2)/Z)다. W를 모르면
이 값을 정할 수 없어서, 지금까지는 근거 없는 15도를 써 왔다.

**픽셀 해상도로는 알 수 없다.** 1920x1080은 14인치 노트북일 수도 있고 55인치
키오스크일 수도 있다 — 같은 픽셀 수에 물리 크기가 4배 차이 난다. 그래서
운영체제가 모니터에서 읽어 온 값을 쓴다.

어디서 오나 — 두 경로, 좋은 것부터
----------------------------------
1) **EDID** (WMI의 WmiMonitorBasicDisplayParams). 모니터가 자기 화면의 실제
   가로·세로를 cm 단위로 들고 있고, 윈도우가 그걸 그대로 넘겨준다. 제조사가
   써 넣은 값이라 가장 믿을 만하다.
2) **GetDeviceCaps(HORZSIZE/VERTSIZE)**. EDID를 못 읽을 때의 차선책인데,
   드라이버가 종종 "96 DPI라고 치고" 계산한 가짜 값을 돌려준다. 그래서
   1번이 실패했을 때만 쓰고, 결과가 의심스러우면 버린다.

둘 다 안 되면 None을 돌려준다 — **지어내지 않는다.** 그러면 설정 파일의
값을 쓰고, 그것도 없으면 예전처럼 고정 각도로 돈다.

데스크탑과 키오스크를 같은 빌드로
---------------------------------
이 자동 인식이 있어야 한 빌드로 둘 다 된다. 14인치 노트북(310mm)과 32인치
키오스크(700mm)는 화면 절반 폭이 2.3배 차이라, 같은 각도 설정을 쓰면 한쪽은
반드시 틀린다. 화면 크기만 자동으로 읽히면 **설치할 때 정할 것은 "그 앞에
앉거나 서는 거리" 하나**로 줄어든다.
"""
import functools
import sys

from src.utils.logger import get_logger

logger = get_logger("utils")

# 말이 되는 범위 — 밖이면 드라이버가 지어낸 값으로 보고 버린다.
# 아래쪽은 10인치 태블릿(가로 약 220mm), 위쪽은 85인치 대형 사이니지(약 1880mm)
MIN_WIDTH_MM = 150.0
MAX_WIDTH_MM = 2200.0
MIN_HEIGHT_MM = 90.0
MAX_HEIGHT_MM = 1400.0
# 가로세로비가 이 범위를 벗어나면 이상하다 (4:3=1.33 ~ 21:9=2.33, 여유 포함).
# 이 검사는 **패널이 원래 생긴 모양** 기준이다 — 세로로 돌려 단 화면이라도
# EDID는 가로형 값을 주므로 여기서는 걸리지 않는다. 돌려 단 것을 실제 화면
# 방향으로 바꾸는 일은 검사를 통과한 뒤 _apply_screen_rotation이 한다
MIN_ASPECT = 1.0
MAX_ASPECT = 3.2


def _sane(width_mm, height_mm):
    """말이 되는 값인가. 아니면 안 쓴다."""
    if not width_mm or not height_mm:
        return False
    if not (MIN_WIDTH_MM <= width_mm <= MAX_WIDTH_MM):
        return False
    if not (MIN_HEIGHT_MM <= height_mm <= MAX_HEIGHT_MM):
        return False
    aspect = width_mm / height_mm
    return MIN_ASPECT <= aspect <= MAX_ASPECT


def _from_edid():
    """모니터가 들고 있는 실제 치수 (WMI). 못 읽으면 None."""
    if sys.platform != "win32":
        return None
    try:
        import subprocess
    except ImportError:
        return None
    try:
        # WMI를 파이썬 의존성 없이 읽는다 — powershell 한 줄.
        # cm 단위 정수라 10을 곱해 mm로 만든다
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-CimInstance -Namespace root\\wmi -ClassName "
             "WmiMonitorBasicDisplayParams | Select-Object -First 1 "
             "-Property MaxHorizontalImageSize,MaxVerticalImageSize | "
             "ForEach-Object { \"$($_.MaxHorizontalImageSize),"
             "$($_.MaxVerticalImageSize)\" })"],
            capture_output=True, text=True, timeout=8.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    if "," not in text:
        return None
    try:
        w_cm, h_cm = (int(v.strip()) for v in text.split(",", 1))
    except ValueError:
        return None
    return (w_cm * 10.0, h_cm * 10.0)


def _from_device_caps():
    """드라이버가 말하는 치수. EDID를 못 읽었을 때만 쓴다."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
    except ImportError:
        return None
    HORZSIZE, VERTSIZE = 4, 6
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        dc = user32.GetDC(0)
        if not dc:
            return None
        try:
            width_mm = float(gdi32.GetDeviceCaps(dc, HORZSIZE))
            height_mm = float(gdi32.GetDeviceCaps(dc, VERTSIZE))
        finally:
            user32.ReleaseDC(0, dc)
    except (AttributeError, OSError):
        return None
    return (width_mm, height_mm)


def _screen_pixel_size():
    """지금 화면의 픽셀 해상도. 화면이 돌아가 있는지 알아보는 데만 쓴다."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
    except ImportError:
        return None
    SM_CXSCREEN, SM_CYSCREEN = 0, 1
    try:
        user32 = ctypes.windll.user32
        width_px = int(user32.GetSystemMetrics(SM_CXSCREEN))
        height_px = int(user32.GetSystemMetrics(SM_CYSCREEN))
    except (AttributeError, OSError):
        return None
    if width_px <= 0 or height_px <= 0:
        return None
    return (width_px, height_px)


def _apply_screen_rotation(width_mm, height_mm):
    """화면을 세로로 돌려 놨으면 가로·세로 mm를 바꿔 준다 (2026-09-09 신설).

    **9:16 세로 키오스크에서 꼭 필요하다.** EDID는 패널이 원래 어떻게 생겼는지를
    말한다 — 가로형 패널을 벽에 세로로 돌려 달아도 EDID는 여전히
    "597 x 336 mm"라고 답한다. 그런데 사용자 앞에 있는 화면은 336mm 폭에
    597mm 높이다. 그대로 쓰면 화면 끝을 보는 고개 각도의 가로·세로가 통째로 뒤바뀌어,
    좌우는 너무 많이 가고 위아래는 모자라게 된다.

    윈도우 회전 설정을 따로 묻지 않고 **픽셀 해상도와 비교**한다. 회전은
    해상도에 그대로 나타나고(1080x1920), 실제 화면과 픽셀의 가로세로 방향은
    반드시 같기 때문이다 — 화면 설정과 드라이버 중 어느 쪽이 회전을 반영하고
    어느 쪽이 안 하든 이 비교는 성립한다.

    두 방향이 확실히 어긋날 때만 바꾼다. 정사각형에 가까운 화면(1.0 근처)에서는
    어느 쪽도 아니므로 손대지 않는다.
    """
    pixels = _screen_pixel_size()
    if not pixels:
        return (width_mm, height_mm)
    pixel_is_portrait = pixels[1] > pixels[0] * 1.05
    mm_is_portrait = height_mm > width_mm * 1.05
    mm_is_landscape = width_mm > height_mm * 1.05
    if pixel_is_portrait and mm_is_landscape:
        logger.info("화면이 세로로 돌아가 있습니다(%dx%d px) - 실치수도 "
                    "%.0f x %.0f mm 로 바꿔 씁니다", pixels[0], pixels[1],
                    height_mm, width_mm)
        return (height_mm, width_mm)
    if not pixel_is_portrait and mm_is_portrait:
        logger.info("화면은 가로인데(%dx%d px) 실치수가 세로로 왔습니다 - "
                    "%.0f x %.0f mm 로 바꿔 씁니다", pixels[0], pixels[1],
                    height_mm, width_mm)
        return (height_mm, width_mm)
    return (width_mm, height_mm)


@functools.lru_cache(maxsize=1)
def detect_screen_size_mm():
    """화면의 실제 (가로mm, 세로mm). 못 알아내면 None.

    한 번만 실제로 알아본다(lru_cache) — 모니터를 바꿔 끼우는 일은 드물고,
    EDID를 읽는 데 시간이 걸려서 매번 부르면 시작이 느려진다.

    지어내지 않는다 — 못 알아내면 None을 주고, 부르는 쪽이 설정값이나
    예전 방식으로 넘어간다.
    """
    for source, fn in (("모니터 EDID", _from_edid),
                       ("그래픽 드라이버", _from_device_caps)):
        try:
            size = fn()
        except Exception:                  # 인식 실패로 프로그램이 죽으면 안 된다
            size = None
        if size and _sane(size[0], size[1]):
            logger.info("화면 물리 크기를 %s에서 읽었습니다: %.0f x %.0f mm",
                        source, size[0], size[1])
            # 세로로 돌려 단 화면이면 여기서 가로·세로를 바꾼다
            # (_apply_screen_rotation 독스트링 참고 — 9:16 키오스크에 필요)
            return _apply_screen_rotation(size[0], size[1])
        if size:
            logger.info("%s가 준 화면 크기(%.0f x %.0f mm)가 말이 안 돼 버립니다",
                        source, size[0], size[1])
    logger.info("화면 물리 크기를 알아내지 못했습니다 - 설정값을 씁니다")
    return None
