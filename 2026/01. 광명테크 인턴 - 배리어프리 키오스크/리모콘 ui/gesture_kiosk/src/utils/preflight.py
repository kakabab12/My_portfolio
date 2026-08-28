"""시작 전 자가 점검 — 문제가 있으면 "무엇이 잘못됐고 어떻게 고치는지"를 알려준다.

왜 필요한가
-----------
키오스크는 개발자가 아닌 사람이 설치하고, 문제가 생겨도 곁에 아무도 없다.
그런데 지금까지는 뭔가 빠지면 이런 메시지만 나왔다:

    KeyError: 'sensitivity_x'
    FileNotFoundError: models/weights/face_landmarker.task

이걸 보고 무엇을 고쳐야 할지 알 수 있는 사람은 이 코드를 쓴 사람뿐이다.
게다가 오류가 하나 날 때마다 프로그램이 죽으니, 고치고 다시 켜고 또 죽고를
반복해야 한다.

그래서 시작할 때 **한 번에 전부 점검하고, 문제를 모아서** 보여준다. 각 문제마다
"어떻게 고치는지"를 같이 적는다. 고칠 수 있는 것과 못 고치는 것을 나눠서,
치명적인 것만 실행을 막는다.
"""
import os

from src.utils.logger import get_logger

logger = get_logger("preflight")

# 프로그램이 실제로 읽는 설정 키들. 하나라도 없으면 나중에 엉뚱한 데서 죽는다.
# (경로, 이 값이 무엇인지) — 사람이 읽을 설명을 같이 둔다
REQUIRED_KEYS = [
    ("camera.device_id", "카메라 장치 번호"),
    ("camera.width_px", "카메라 가로 해상도"),
    ("camera.height_px", "카메라 세로 해상도"),
    ("face_tracker.model_path", "얼굴 인식 모델 파일 경로"),
    ("face_tracker.max_num_faces", "동시에 찾을 얼굴 수"),
    ("head_tracker.calibration_window_sec", "처음 자세를 재는 시간"),
    ("head_tracker.pointer.sensitivity_x", "커서 좌우 감도"),
    ("head_tracker.pointer.sensitivity_y", "커서 상하 감도"),
    ("head_tracker.pointer.smoothing_alpha", "커서 부드러움 정도"),
    ("head_tracker.pointer.max_offset_ratio", "커서가 갈 수 있는 최대 범위"),
    ("head_tracker.mouth_click.open_margin", "입을 벌렸다고 볼 기준"),
    ("head_tracker.mouth_click.close_margin", "입을 다물었다고 볼 기준"),
    ("head_tracker.dwell_click.dwell_sec", "응시 클릭까지 걸리는 시간"),
    ("logging.save_dir", "로그를 저장할 폴더"),
]

MODEL_MIN_BYTES = 1_000_000     # 정상 모델은 수 MB — 이보다 작으면 받다 만 파일이다


class Problem:
    """점검에서 발견한 문제 하나. fatal이면 실행을 막는다."""

    def __init__(self, what, how, fatal=True):
        self.what = what      # 무엇이 잘못됐나
        self.how = how        # 어떻게 고치나
        self.fatal = fatal


def _dig(config, dotted):
    """'a.b.c' 경로로 설정값을 꺼낸다. 없으면 KeyError."""
    node = config
    for part in dotted.split("."):
        node = node[part]
    return node


def check_config(config):
    """설정에 필요한 값이 다 있는지, 값이 말이 되는지 본다."""
    problems = []
    for dotted, meaning in REQUIRED_KEYS:
        try:
            _dig(config, dotted)
        except (KeyError, TypeError):
            problems.append(Problem(
                "설정에 '%s' 항목이 없습니다 (%s)" % (dotted, meaning),
                "configs/config.yaml 을 열어 해당 항목을 추가하세요. "
                "지운 적이 없다면 저장소의 원본 config.yaml과 비교해 보세요."))

    # 값이 있어도 말이 안 되는 경우를 걸러낸다 — 이런 건 실행은 되는데
    # 동작이 이상해서 원인을 찾기가 훨씬 어렵다
    checks = [
        ("head_tracker.pointer.sensitivity_x", 0.05, 10.0, "커서 좌우 감도"),
        ("head_tracker.pointer.sensitivity_y", 0.05, 10.0, "커서 상하 감도"),
        ("head_tracker.pointer.smoothing_alpha", 0.01, 1.0, "커서 부드러움 정도"),
        ("head_tracker.pointer.max_offset_ratio", 0.05, 1.0, "커서 최대 범위"),
        ("head_tracker.calibration_window_sec", 0.05, 30.0, "자세 재는 시간"),
        ("face_tracker.max_num_faces", 1, 10, "동시에 찾을 얼굴 수"),
    ]
    for dotted, lo, hi, meaning in checks:
        try:
            v = float(_dig(config, dotted))
        except Exception:
            continue          # 없는 건 위에서 이미 잡았다
        if not (lo <= v <= hi):
            problems.append(Problem(
                "'%s' 값이 %s입니다 — %s는 보통 %s ~ %s 사이입니다"
                % (dotted, v, meaning, lo, hi),
                "configs/config.yaml 에서 이 값을 범위 안으로 고치세요.",
                fatal=False))
    return problems


def check_model(config):
    """얼굴 인식 모델 파일이 제자리에 있고 온전한지 본다."""
    try:
        path = _dig(config, "face_tracker.model_path")
    except Exception:
        return []             # 경로 자체가 없는 건 check_config이 잡는다
    if not os.path.exists(path):
        return [Problem(
            "얼굴 인식 모델 파일이 없습니다: %s" % path,
            "scripts/download_weights.py 를 실행해 모델을 내려받으세요. "
            "네트워크가 안 되는 곳이면 다른 PC에서 받아 models/weights/ 에 복사해도 됩니다.")]
    size = os.path.getsize(path)
    if size < MODEL_MIN_BYTES:
        return [Problem(
            "모델 파일이 너무 작습니다 (%.1fMB) — 받다가 끊긴 것으로 보입니다: %s"
            % (size / 1e6, path),
            "그 파일을 지우고 scripts/download_weights.py 로 다시 받으세요.")]
    return []


def check_camera(config, quick=True):
    """카메라가 열리는지 본다. quick이면 열어보고 바로 닫는다(시작 지연 최소화)."""
    try:
        device_id = int(_dig(config, "camera.device_id"))
    except Exception:
        return []
    try:
        import cv2
        backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(device_id, backend)
        opened = cap.isOpened()
        got_frame = False
        if opened and not quick:
            got_frame, _ = cap.read()
        cap.release()
    except Exception as exc:
        return [Problem(
            "카메라를 확인하는 중 오류가 났습니다: %s" % exc,
            "카메라가 연결돼 있는지, 다른 프로그램이 쓰고 있지 않은지 확인하세요.")]

    if not opened:
        return [Problem(
            "%d번 카메라를 열 수 없습니다" % device_id,
            "① 카메라 USB 연결 확인  ② 다른 프로그램(화상회의 등)이 쓰고 있는지 확인  "
            "③ 카메라가 여러 대면 configs/config.yaml 의 camera.device_id 를 바꿔 보세요.")]
    if not quick and not got_frame:
        return [Problem(
            "%d번 카메라는 열렸지만 화면이 들어오지 않습니다" % device_id,
            "USB를 뽑았다 다시 꽂아 보세요. 그래도 안 되면 다른 USB 단자에 연결해 보세요.")]
    return []


def run(config, check_cam=True):
    """전부 점검해 문제 목록을 돌려준다. 문제가 없으면 빈 목록."""
    problems = []
    problems += check_config(config)
    problems += check_model(config)
    if check_cam:
        problems += check_camera(config)
    return problems


def report(problems):
    """사람이 읽을 형태로 찍는다. 치명적인 문제가 하나라도 있으면 True."""
    if not problems:
        logger.info("시작 전 점검 통과 — 설정·모델·카메라 모두 정상")
        return False

    fatal = [p for p in problems if p.fatal]
    warn = [p for p in problems if not p.fatal]

    lines = ["", "=" * 66, "시작 전 점검에서 문제를 찾았습니다", "=" * 66]
    for i, p in enumerate(fatal, 1):
        lines += ["", "[실행 불가 %d] %s" % (i, p.what), "   해결: %s" % p.how]
    for i, p in enumerate(warn, 1):
        lines += ["", "[확인 필요 %d] %s" % (i, p.what), "   해결: %s" % p.how]
    lines += ["", "=" * 66, ""]
    text = "\n".join(lines)
    print(text)
    logger.error("시작 전 점검 실패 — 실행 불가 %d건, 확인 필요 %d건",
                 len(fatal), len(warn))
    return bool(fatal)
