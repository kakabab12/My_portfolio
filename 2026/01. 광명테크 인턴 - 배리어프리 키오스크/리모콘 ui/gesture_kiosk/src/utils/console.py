"""콘솔 출력이 프로그램을 멈추지 못하게 한다 (2026-08-25 신설).

왜 필요한가
-----------
head.py·eyebrow.py는 매 프레임 커서 좌표를 한 줄씩 찍는다(main_dpad.py와
공유하는 콘솔 규약 — 클릭·드래그 발화도 한 줄씩 찍는다). 콘솔 창에 직접
띄우면 아무 문제가 없다. 실측 0.002ms로 사실상 공짜다.

문제는 **다른 프로그램이 이 프로그램을 띄우면서 출력을 파이프로 받는**
경우다 — 런처, 서비스 래퍼, 작업 스케줄러 등이 그렇게 한다. 받아만 놓고
읽지 않으면 파이프 버퍼가 가득 차고, 그 다음 print가 **영원히 반환하지
않는다**. 추론 스레드가 거기서 멈춰 얼굴 인식이 통째로 정지한다. 그런데
프로그램은 살아 있고 오류도 안 나서, 밖에서 보면 "이유 없이 멈췄다"로만
보인다 — 원인을 찾기가 대단히 어려운 부류의 고장이다.

실측(2026-08-25): 아무도 읽지 않는 파이프에 연결하니 **8.2초 / 240줄** 만에
완전히 멈췄다. 윈도우 파이프 버퍼가 작아서(4KB 남짓) 생각보다 훨씬 빨리 찬다.

어떻게 푸는가
-------------
출력을 전담하는 데몬 스레드를 하나 두고, 본 스레드는 큐에 넣기만 하고
곧바로 돌아온다. 큐가 차면 **넣지 않고 버린다** — 막히느니 몇 줄 잃는 게
낫다. 읽는 쪽이 정상이면 예전과 똑같이 전부 나가고(규약이 그대로 지켜진다),
막히면 출력 스레드 혼자 멈춰 있을 뿐 추론과 렌더는 계속 돈다.

버린 줄 수는 세어 두었다가 종료할 때 한 번 기록한다 — "출력이 좀 빠졌다"는
사실 자체를 나중에 알 수 있어야 하기 때문이다.
"""
import queue
import sys
import threading

from src.utils.logger import get_logger

logger = get_logger("console")

# 큐 길이 — 30fps 기준 약 30초치. 읽는 쪽이 잠깐 느려지는 정도는 다 흡수하고,
# 아예 안 읽으면 이만큼만 쌓였다가 그 뒤로는 버린다
MAX_PENDING_LINES = 1000


def enable_utf8_output():
    """한글·특수문자를 print해도 죽지 않게 표준 출력을 UTF-8로 맞춘다.

    ★2026-08-28 신설 — **같은 버그에 네 번째로 걸려서** 공용 함수로 만들었다.

    [무슨 버그인가]
    한국어 윈도우의 기본 콘솔 인코딩은 cp949인데, 여기엔 줄표(—, U+2014)를
    담을 자리가 없다. 그래서 `print("... — ...")` 한 줄이 UnicodeEncodeError로
    **프로그램을 통째로 죽인다.** 정작 하려던 일은 다 끝내 놓고 마지막
    안내문을 찍다가 죽는 경우가 많아, 겉보기엔 "성공했는데 오류가 났다"로
    보인다.

    [왜 반복됐나]
    2026-08-25 트래커 본체에서 고치고, 08-27 빌드 스크립트 4개에서 또 고치고,
    같은 날 변환 스크립트에서 또 겪었다. 매번 **그 파일에만** 인라인으로
    고쳤기 때문에, 새 스크립트를 만들면 그대로 재발했다(08-28
    measure_head_pose.py). 파일마다 고치는 대신 여기 한 곳에 두고 부르게 한다.

    [쓰는 법] 새 스크립트를 만들면 main() 맨 앞에서 한 번 부른다.

        from src.utils.console import enable_utf8_output
        enable_utf8_output()

    콘솔이 아예 없는 환경(pythonw, 서비스)에서도 안전하다 — 조용히 넘어간다.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                # 진짜 콘솔 — 인코딩은 콘솔이 정하게 두고, 못 그리는 글자만
                # 대체 문자로 바꾼다(코드페이지를 강제로 바꾸면 다른 프로그램
                # 출력까지 영향을 받는 경우가 있다)
                reconfigure(errors="replace")
            else:
                # 파이프·파일로 나가는 중 — UTF-8로 고정하는 게 안전하다
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:   # noqa: 방어적 — 인코딩 설정 실패로 본 기능이 죽으면 안 된다
            pass


class _ConsoleWriter:
    def __init__(self):
        self._queue = queue.Queue(maxsize=MAX_PENDING_LINES)
        self._thread = None
        self._dropped = 0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def emit(self, line):
        """한 줄 찍기를 예약한다. 절대 막히지 않는다."""
        if self._thread is None:
            # 아직 안 켰으면(테스트 등) 예전처럼 그냥 찍는다 — 이 경로는
            # 스레드가 하나뿐인 상황이라 막힐 일이 없다
            print(line)
            return
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._dropped += 1

    def _loop(self):
        while True:
            line = self._queue.get()
            if line is None:      # stop() 신호
                return
            try:
                print(line)
            except Exception:     # noqa: 방어적 — 출력이 깨져도 본 기능은 계속
                pass

    def stop(self):
        """남은 줄을 내보내고 멈춘다. 읽는 쪽이 막혀 있으면 기다리지 않는다."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)
        self._thread = None
        if self._dropped:
            logger.warning(
                "콘솔 출력 %d줄을 건너뛰었습니다 — 출력을 받아가는 쪽이 "
                "읽지 않아 밀렸습니다(프로그램 동작에는 영향 없음)", self._dropped)


_writer = _ConsoleWriter()

start = _writer.start
emit = _writer.emit
stop = _writer.stop


def dropped_count():
    """지금까지 버린 줄 수 — 시험·진단용."""
    return _writer._dropped
