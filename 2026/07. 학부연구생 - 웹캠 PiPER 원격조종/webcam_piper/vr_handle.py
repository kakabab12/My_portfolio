"""VR 손잡이로 PiPER 조종 — 헤드셋은 머리에 쓰지 않고 옆에 세워 둔다 (2026-11-17).

사용자 (11/17): "손으로 말고 VR 핸들만 잡고, 머리에 착용하는 건 제외하고",
"로봇팔 그리퍼를 A4 십자 가운데 두고 컨트롤러도 똑같이 둬서 기준을 잡고 움직이게".

- Quest 컨트롤러 위치는 헤드셋 카메라가 잰다. 그래서 헤드셋을 머리 높이쯤 선반·삼각대에 손 쪽을 내려다보게 두고,
  렌즈 사이 근접 센서를 테이프로 가린다 (안 가리면 벗은 뒤 잠들어 추적이 멈춤).
- 이 PC 가 https 로 조종 화면을 내준다. Quest 브라우저에서 https://<이 PC 주소>:8443 → '안전하지 않음' 경고에서 계속 → Start.
  Node·npm 필요 없음. 찬혁님 piper-teleop 의 웹 앱·서버는 건드리지 않는다 (그쪽은 카메라 2대가 꼭 있어야 켜지고,
  쥐기 버튼을 떼면 관절 0 자세로 돌아감).
- 기준 (웹캠 조종과 같은 규칙): 사람 쪽 A4 + 에 컨트롤러 앞 끝을 대고, 앞(종이 먼 쪽)을 가리킨 채 B 를 1초 누른다.
  그 자리 = 로봇 쪽 A4 + (밑동 앞 x0, 옆 y0) 의 종이 면 (+ lift). 그 뒤 컨트롤러가 움직인 만큼 집게 끝이 똑같이 움직인다.
  그래서 두 책상에서 각자 + 기준으로 같은 자리에 물체를 두면, 컨트롤러로 집는 자리에 집게도 간다.
  로봇 집게를 직접 + 에 옮겨 놓을 필요는 없다 (로봇 종이를 밑동 앞 30 cm 에 두면 계산으로 앎).

오른손 컨트롤러
  쥐기 버튼(가운데 손가락) 누르고 있음 = 따라감. 처음엔 천천히 다가가고(손 멈춰 두기), 맞춰지면 짧게 두 번 떨림 → 그때부터 같은 속도
  쥐기 버튼 놓음 = 그 자리에 멈춤 (처음 자세로 돌아가지 않음)
  방아쇠(검지) = 집게. 당길수록 닫힘
  B 1초 = 기준 잡기 (짧게 세 번 떨림)      A = 기록 시작 (한 번 떨림) / 저장 (두 번)
  추적이 끊기거나(헤드셋이 컨트롤러를 못 봄·잠듦) 신호가 0.25초 넘게 안 오면 멈추고 길게 떨림. 다시 하려면 쥐기 버튼을 놓았다 누름.

실행 (찬혁님 .venv: numpy·websockets·agx-bus. 웹캠 쪽과 같은 폴더의 mapping·robots·piper_kin 을 씀)
  python vr_handle.py --selftest                                 # Quest 없이 가짜 컨트롤러로 계산 길 시험 (팔 없음)
  python vr_handle.py                                            # 가상 팔: Quest 연결·추적만 시험
  python vr_handle.py --robot agxbus --speed 50 --lift 0.10      # 실물 첫 시험: 집게가 종이 위 10 cm 떠서 따라감
  python vr_handle.py --robot agxbus --speed 100                 # 사람과 같은 속도
  python vr_handle.py --report data/날짜_시각/session.npz        # 끝난 뒤: 늦음·속도·위아래 숫자
"""
import argparse
import asyncio
import collections
import itertools
import json
import math
import os
import ssl
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mapping import Mapper, wrap                  # noqa: E402
from net import my_ips                            # noqa: E402
from piper_kin import TABLE_Z, PiperKin, grip_orientation   # noqa: E402
from recorder import Recorder                     # noqa: E402
from robots import GRIP_MAX, make_arm             # noqa: E402

PORT = 8443
# WebXR (x 오른쪽, y 위, z 뒤 = 사람 쪽) → 로봇 축 (x 앞, y 왼쪽, z 위). 찬혁님 quest_to_flange 와 같은 바꿈
X2R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
STATE_CODE = {"idle": 0, "catchup": 1, "follow": 2}
HAPTIC = {"short": [(0.0, 0.6, 60)], "double": [(0.0, 0.8, 60), (0.15, 0.8, 60)],
          "triple": [(0.0, 0.8, 50), (0.13, 0.8, 50), (0.26, 0.8, 50)], "long": [(0.0, 1.0, 450)]}

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PiPER VR handle</title>
<style>body{font-family:sans-serif;background:#15181c;color:#e8e8e8;margin:0;padding:28px;font-size:22px;line-height:1.45}
button{font-size:28px;padding:18px 30px;margin:10px 14px 10px 0;border-radius:12px;border:0;background:#3d7bd9;color:#fff}
button.alt{background:#555}#st{margin:18px 0;padding:14px;background:#22272e;border-radius:10px;white-space:pre-wrap}
ol{padding-left:28px}small{color:#aaa}</style></head><body>
<h2>PiPER VR handle</h2>
<div id="st">connecting to robot PC...</div>
<button id="ar">Start (passthrough)</button><button id="vr" class="alt">Start (VR)</button>
<ol><li>Tape over the proximity sensor (between the lenses).</li>
<li>Press Start, then put the headset on a shelf at head height, looking down at your hands.</li>
<li>Controller tip on the paper cross, pointing forward: hold <b>B</b> 1 s (3 buzzes).</li>
<li>Hold <b>grip</b> to follow (hold still until 2 buzzes). <b>Trigger</b> = gripper. <b>A</b> = record.</li></ol>
<small>근접 센서 가리기 → Start → 헤드셋을 머리 높이 선반에 → 종이 + 에서 B 1초 → 쥐기 버튼 누른 채 움직임</small>
<script>
const S = document.getElementById('st');
let ws = null, gl = null, ref = null, seq = 0, haptic = [], inXR = false, last = '';
function show(t){ S.textContent = t; }
function connect(){
  ws = new WebSocket('wss://' + location.host + '/ws');
  ws.onopen = () => show('robot PC connected. Press Start.');
  ws.onclose = () => { show('robot PC disconnected - retrying...'); setTimeout(connect, 1000); };
  ws.onmessage = (e) => { try { const m = JSON.parse(e.data);
      if (m.hap) haptic.push(m.hap);
      if (m.txt) { last = m.txt; if (!inXR) show(m.txt); } } catch (_) {} };
}
connect();
async function start(mode){
  if (!navigator.xr) { show('WebXR not available. Open this page with https://'); return; }
  if (!(await navigator.xr.isSessionSupported(mode))) { show(mode + ' not supported here'); return; }
  const s = await navigator.xr.requestSession(mode, {optionalFeatures: ['local-floor']});
  const c = document.createElement('canvas');
  gl = c.getContext('webgl', {xrCompatible: true});
  s.updateRenderState({baseLayer: new XRWebGLLayer(s, gl)});
  try { ref = await s.requestReferenceSpace('local-floor'); } catch (_) { ref = await s.requestReferenceSpace('local'); }
  inXR = true;
  s.addEventListener('end', () => { inXR = false; show('XR ended. Press Start again.\n' + last); });
  s.requestAnimationFrame(frame);
}
function frame(t, f){
  const s = f.session; s.requestAnimationFrame(frame);
  const L = s.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, L.framebuffer); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  const m = {seq: ++seq, t: Math.round(t * 10) / 10, vis: s.visibilityState};
  for (const src of s.inputSources) {
    if (src.handedness !== 'right' || !src.gamepad) continue;
    const b = src.gamepad.buttons;
    m.tr = b[0] ? +b[0].value.toFixed(3) : 0;
    m.sq = b[1] ? +b[1].value.toFixed(3) : 0;
    m.a = (b[4] && b[4].pressed) ? 1 : 0;
    m.b = (b[5] && b[5].pressed) ? 1 : 0;
    const pose = f.getPose(src.targetRaySpace, ref);
    if (pose) { const p = pose.transform.position, o = pose.transform.orientation;
      m.p = [p.x, p.y, p.z].map(v => +v.toFixed(5)); m.q = [o.x, o.y, o.z, o.w].map(v => +v.toFixed(5));
      m.emu = pose.emulatedPosition ? 1 : 0; }
    if (haptic.length) { const h = haptic.shift(); try {
        const ha = src.gamepad.hapticActuators;
        if (ha && ha.length) ha[0].pulse(h[0], h[1]);
        else if (src.gamepad.vibrationActuator) src.gamepad.vibrationActuator.playEffect('dual-rumble',
            {duration: h[1], strongMagnitude: h[0], weakMagnitude: h[0]}); } catch (_) {} }
  }
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(m));
}
document.getElementById('ar').onclick = () => start('immersive-ar').catch(e => show('XR failed: ' + e));
document.getElementById('vr').onclick = () => start('immersive-vr').catch(e => show('XR failed: ' + e));
</script></body></html>
"""


def quat_mat(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading(Rr):
    """컨트롤러가 가리키는 방향(레이 −Z)의 수평 각 (로봇 축, rad). 거의 위·아래를 가리키면 컨트롤러 오른쪽(+X) 으로 잰다."""
    f = Rr @ np.array([0.0, 0.0, -1.0])
    if math.hypot(f[0], f[1]) > 0.35:
        return math.atan2(f[1], f[0])
    r = Rr @ np.array([1.0, 0.0, 0.0])
    return math.atan2(r[1], r[0]) + math.pi / 2


def ensure_cert(folder):
    """https 용 자체 서명 인증서 (처음 한 번, 이 PC 주소가 바뀌면 다시). 폴더는 .gitignore 에 있음."""
    os.makedirs(folder, exist_ok=True)
    cert, key, meta = (os.path.join(folder, n) for n in ("cert.pem", "key.pem", "san.json"))
    ips = sorted(set(my_ips()) | {"127.0.0.1"})
    try:
        with open(meta, encoding="utf-8") as f:
            old = set(json.load(f))
    except (OSError, ValueError):
        old = None
    if not (os.path.exists(cert) and os.path.exists(key) and old is not None and set(ips) <= old):
        san = ",".join([f"IP:{i}" for i in ips] + ["DNS:localhost"])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                        "-days", "825", "-subj", "/CN=piper-vr-handle", "-addext", f"subjectAltName={san}"],
                       check=True, capture_output=True)
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(ips, f)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


class QuestLink:
    """Quest 조종 화면과의 연결. asyncio 를 따로 스레드에서 돌린다. 가장 새로 붙은 화면 하나만 듣는다."""

    def __init__(self, port, ssl_ctx):
        self.port, self.ssl_ctx = port, ssl_ctx
        self.page = PAGE.encode("utf-8")
        self.lock = threading.Lock()
        self.msg, self.rx_t, self.n_rx, self.gen, self.active, self.peer = None, None, 0, 0, None, None
        self.outbox = collections.deque(maxlen=64)
        self.ids = itertools.count(1)
        self.loop = self.stop_ev = None
        self.error = None
        self.ready = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        self.ready.wait(10)

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # noqa: BLE001 — 포트가 쓰이는 중 등
            self.error = e
            self.ready.set()

    async def _main(self):
        from websockets.asyncio.server import serve
        self.loop = asyncio.get_running_loop()
        self.stop_ev = asyncio.Event()
        async with serve(self._handler, "0.0.0.0", self.port, ssl=self.ssl_ctx, process_request=self._http,
                         max_size=2 ** 16, ping_interval=None, compression=None):
            self.ready.set()
            await self.stop_ev.wait()

    def _http(self, conn, req):
        if req.path.startswith("/ws"):
            return None                                       # 웹소켓으로 이어감
        from websockets.datastructures import Headers
        from websockets.http11 import Response
        if req.path.split("?")[0] in ("/", "/index.html"):
            return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                ("Content-Length", str(len(self.page))),
                                                ("Cache-Control", "no-store")]), self.page)
        return conn.respond(404, "not found\n")

    async def _handler(self, conn):
        cid = next(self.ids)
        with self.lock:
            self.active, self.gen, self.msg, self.rx_t = cid, self.gen + 1, None, None
            try:
                self.peer = conn.remote_address[0]
            except Exception:  # noqa: BLE001
                self.peer = "?"
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(conn.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    raw = None
                with self.lock:
                    if self.active != cid:
                        break                                 # 더 새 화면이 붙음
                    if raw is not None:
                        try:
                            self.msg, self.rx_t = json.loads(raw), time.monotonic()
                            self.n_rx += 1
                        except ValueError:
                            pass
                    out = list(self.outbox)
                    self.outbox.clear()
                for m in out:
                    await conn.send(json.dumps(m))
        except Exception:  # noqa: BLE001 — 연결 끊김
            pass
        finally:
            with self.lock:
                if self.active == cid:
                    self.active = None

    def snapshot(self):
        with self.lock:
            return self.msg, self.rx_t, self.n_rx, self.gen, self.active is not None, self.peer

    def send(self, **m):
        with self.lock:
            self.outbox.append(m)

    def close(self):
        if self.loop is not None and self.stop_ev is not None:
            self.loop.call_soon_threadsafe(self.stop_ev.set)
        self.th.join(2.0)


def parse(argv=None):
    ap = argparse.ArgumentParser(description="VR 손잡이 → PiPER (헤드셋은 옆에 세워 둠)")
    ap.add_argument("--robot", choices=["preview", "agxbus", "sdk"], default="preview")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--speed", type=int, default=50, help="실물 팔 속도 %% (사람과 같은 속도는 100)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--mode", choices=["paper", "clutch"], default="paper",
                    help="paper: 종이 + 기준(B 1초)으로 같은 자리 (기본). clutch: 쥐기 버튼 누른 자리에서부터 움직인 만큼")
    ap.add_argument("--x0", type=float, default=0.30, help="로봇 쪽 종이 + : 밑동에서 앞으로 (m)")
    ap.add_argument("--y0", type=float, default=0.0, help="로봇 쪽 종이 + : 밑동에서 왼쪽으로 (m)")
    ap.add_argument("--lift", type=float, default=0.0, help="집게를 이만큼(m) 띄워서 따라감. 실물 첫 시험은 0.10")
    ap.add_argument("--gain", type=float, default=1.0, help="컨트롤러 1 cm → 집게 몇 cm")
    ap.add_argument("--min-z", type=float, default=0.01, help="집게 끝을 책상 위 이 높이(m) 아래로 내리지 않음")
    ap.add_argument("--max-z", type=float, default=0.35, help="집게 끝을 책상 위 이 높이(m) 위로 올리지 않음")
    ap.add_argument("--max-speed", type=float, default=1.0, help="집게 끝 최고 속도 (m/s). 넘으면 측정 튐으로 보고 자름")
    ap.add_argument("--catchup-speed", type=float, default=0.08, help="따라가기 시작할 때 다가가는 속도 (m/s)")
    ap.add_argument("--rot-rate", type=float, default=180.0, help="집게 방향 최고 도는 속도 (도/s)")
    ap.add_argument("--joint-rate", type=float, default=3.0, help="관절 최고 속도 (rad/s)")
    ap.add_argument("--no-yaw", action="store_true", help="컨트롤러 돌림을 따라가지 않음 (집게 방향 고정)")
    ap.add_argument("--pull-open", action="store_true", help="방아쇠를 당기면 집게가 열림 (찬혁님 방식). 기본은 당기면 닫힘")
    ap.add_argument("--hz", type=float, default=100.0, help="팔에 명령 보내는 박자")
    ap.add_argument("--timeout", type=float, default=0.25, help="Quest 신호가 이만큼(초) 안 오면 멈춤")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--report", default=None, metavar="session.npz", help="끝난 기록을 숫자로 보기")
    ap.add_argument("--selftest", action="store_true", help="Quest 없이 가짜 컨트롤러로 시험 (가상 팔)")
    ap.add_argument("--run-seconds", type=float, default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


class Controller:
    """Quest 신호 → 기준·따라가기 상태 → Mapper.step_robot → 팔. 박자마다 tick() 을 부른다."""

    def __init__(self, a, link, arm, kin, say=print):
        self.a, self.link, self.arm, self.kin, self.say_ = a, link, arm, kin, say
        self.mapper = Mapper(kin, a.x0, a.y0, a.gain, max_speed=a.max_speed, max_yaw_rate=math.radians(a.rot_rate),
                             max_joint_rate=a.joint_rate, min_z=a.min_z, max_z=a.max_z)
        self.mapper.sync_to(arm.joints())
        self.robot_ref = np.array([a.x0, a.y0, TABLE_Z + a.lift])
        self.state, self.ref, self.anchor = "idle", None, None
        self.need_release, self.sq_on, self.gen_seen = False, False, 0
        self.b_since, self.b_done, self.b_samples, self.a_prev = None, False, [], 0
        self.hd_prev, self.hd_unw = None, 0.0
        self.recent_p = collections.deque(maxlen=60)          # 새 신호 60개(약 0.8초)가 똑같으면 추적이 굳은 것으로 봄
        self.n_rx_seen = 0
        self.haptics = []
        self.rows = []
        self.rec = Recorder(a.out, {"args": vars(a), "source": "vr_handle (Quest 컨트롤러, 헤드셋은 옆에)",
                                    "robot_ref": self.robot_ref.tolist(),
                                    "note": "p: 컨트롤러 앞 끝 위치를 로봇 축으로 바꾼 것 (WebXR 원점 기준, m). "
                                            "target: 매핑한 집게 끝 목표 (작업 상자로 자르기 전), cur: 속도 제한 뒤 목표. "
                                            "로봇 좌표: 밑동 (0,0,0.90), 책상 z 0.90"})
        self.events = []                       # (t, 글) — 자체 시험·기록용
        self.msg_txt = ""
        self.t_prev = time.monotonic()
        self.stat_t, self.stat_n = self.t_prev, 0
        self.last_print = ""

    # ---------- 작은 도구
    def say(self, text, t=None):
        self.events.append((time.monotonic() if t is None else t, text))
        self.msg_txt = text
        self.say_("[VR] " + text)
        self.link.send(txt=text)

    def buzz(self, kind, now):
        for dt, i, ms in HAPTIC[kind]:
            self.haptics.append((now + dt, i, ms))

    def stop(self, why, now, need_release=False, buzz=None):
        if self.state != "idle":
            self.arm.hold()
        self.state = "idle"
        self.need_release = self.need_release or need_release
        if buzz:
            self.buzz(buzz, now)
        if why:
            self.say(why, now)

    def mapped(self, p, hd):
        """컨트롤러 → (집게 끝 목표, 집게 yaw)."""
        if self.a.mode == "paper":
            d = rot_z(-self.ref["h"]) @ (p - self.ref["p"])
            return self.robot_ref + self.a.gain * d, hd - self.ref["hu"]
        an = self.anchor
        h = self.ref["h"] if self.ref else 0.0
        return an["robot"] + self.a.gain * (rot_z(-h) @ (p - an["p"])), an["yaw"] + (hd - an["hu"])

    # ---------- 한 박자
    def tick(self):
        a, mp = self.a, self.mapper
        now = time.monotonic()
        dt, self.t_prev = now - self.t_prev, now
        msg, rx_t, n_rx, gen, connected, peer = self.link.snapshot()
        if gen != self.gen_seen:
            self.gen_seen = gen
            self.stop("Quest page connected" + (f" ({peer})" if peer else "") +
                      (": set the reference again (B 1 s on the paper cross)" if a.mode == "paper" else ""), now)
            self.ref, self.recent_p = None, collections.deque(maxlen=60)
        age = now - rx_t if rx_t is not None else float("inf")
        m = msg or {}
        p = hd = Rr = None
        if "p" in m and "q" in m:
            p = X2R @ np.asarray(m["p"], float)
            Rr = X2R @ quat_mat(m["q"])
            h = heading(Rr)
            self.hd_unw = h if self.hd_prev is None else self.hd_unw + wrap(h - self.hd_prev)
            self.hd_prev, hd = h, self.hd_unw
            if age < a.timeout and n_rx != self.n_rx_seen:      # 새로 온 신호만 (같은 신호를 박자마다 다시 읽음)
                self.recent_p.append(tuple(np.round(p, 6)))
        self.n_rx_seen = n_rx
        frozen = len(self.recent_p) == self.recent_p.maxlen and len(set(self.recent_p)) == 1
        track = p is not None and int(m.get("emu", 1)) == 0 and m.get("vis") == "visible" and age < a.timeout \
            and not frozen
        why_lost = ("no signal from Quest" if age >= a.timeout else
                    "right controller not seen" if p is None else
                    "headset not seeing the controller" if int(m.get("emu", 1)) else
                    "XR paused (headset asleep?)" if m.get("vis") != "visible" else "tracking frozen")
        sq = float(m.get("sq", 0.0)) if age < a.timeout else 0.0
        rise = False
        if self.sq_on and sq < 0.35:
            self.sq_on = False
        elif not self.sq_on and sq > 0.65:
            self.sq_on, rise = True, True
        if not self.sq_on:
            self.need_release = False
        tr = float(m.get("tr", 0.0))
        width = GRIP_MAX * (tr if a.pull_open else 1.0 - tr)

        # B 1초: 기준 잡기 (따라가는 중에는 안 받음)
        if int(m.get("b", 0)) and age < a.timeout:
            if self.b_since is None:
                self.b_since, self.b_done, self.b_samples = now, False, []
            if track:
                self.b_samples.append((now, p, hd))
            if not self.b_done and now - self.b_since >= 1.0:
                self.b_done = True
                pts = [(pp, hh) for tt, pp, hh in self.b_samples if now - tt < 0.6]
                if self.state != "idle":
                    self.say("release the grip button before setting the reference", now)
                    self.buzz("long", now)
                elif len(pts) < 10:
                    self.say(f"reference failed: tracking ({why_lost})", now)
                    self.buzz("long", now)
                else:
                    P = np.array([pp for pp, _ in pts])
                    spread = float(np.max(np.linalg.norm(P - np.median(P, 0), axis=1)))
                    if spread > 0.015:
                        self.say(f"reference failed: hold still ({spread * 100:.1f} cm moved)", now)
                        self.buzz("long", now)
                    else:
                        hu = float(np.median([hh for _, hh in pts]))
                        self.ref = {"p": np.median(P, 0), "hu": hu, "h": wrap(hu), "t": now}
                        self.say(f"reference set: controller = robot paper cross (x0 {a.x0:.2f} y0 {a.y0:.2f}"
                                 f" lift {a.lift:.2f} m), forward = {math.degrees(wrap(hu)):+.0f} deg in XR", now)
                        self.buzz("triple", now)
        else:
            self.b_since = None

        # A: 기록 시작 / 저장
        av = int(m.get("a", 0)) if age < a.timeout else 0
        if av and not self.a_prev:
            if self.rec.on:
                r = self.rec.stop()
                self.say(f"saved {os.path.basename(r[0])} ({r[1]} rows)" if r else "nothing recorded", now)
                self.buzz("double", now)
            else:
                self.rec.start()
                self.say(f"recording ep_{len(self.rec.saved):03d}", now)
                self.buzz("short", now)
        self.a_prev = av

        # 따라가기
        tgt = yaw = None
        if self.state == "idle":
            if rise and not self.need_release:
                if a.mode == "paper" and self.ref is None:
                    self.say("set the reference first: controller tip on the paper cross, hold B 1 s", now)
                    self.buzz("long", now)
                    self.need_release = True
                elif not track:
                    self.say(f"cannot follow: {why_lost}", now)
                    self.buzz("long", now)
                    self.need_release = True
                else:
                    mp.sync_to(self.arm.joints())
                    if a.mode == "clutch":
                        self.anchor = {"p": p.copy(), "hu": hd, "robot": mp.cur.copy(), "yaw": mp.cur_yaw}
                        self.state = "follow"
                        self.say("follow (clutch): moving with the controller", now)
                        self.buzz("double", now)
                    else:
                        self.state = "catchup"
                        self.say("approaching slowly - hold still", now)
                        self.buzz("short", now)
        elif not self.sq_on:
            self.stop("grip released: stopped here", now)
        elif not track:
            self.stop(f"stopped: {why_lost}. release and press grip again", now, need_release=True, buzz="long")
        if self.state in ("catchup", "follow"):
            tgt, yaw = self.mapped(p, hd)
            yaw = None if a.no_yaw else yaw
            if self.state == "catchup":
                q_cmd, w = mp.step_robot(tgt, dt, yaw=yaw, width=width, max_speed=a.catchup_speed,
                                         max_rot_rate=math.radians(45))
                err = float(np.linalg.norm(mp.cur - mp.clip_robot(tgt)))
                yaw_err = abs(mp.yaw_goal - mp.cur_yaw)
                rot_err = math.degrees(math.acos(np.clip((np.trace(grip_orientation(mp.cur_yaw) @ mp.cur_mat.T) - 1) / 2, -1, 1)))
                if err < 0.01 and yaw_err < math.radians(5) and rot_err < 5:
                    self.state = "follow"
                    self.say("matched: now same speed as your hand", now)
                    self.buzz("double", now)
            else:
                q_cmd, w = mp.step_robot(tgt, dt, yaw=yaw, width=width)
            self.arm.send(q_cmd, w)
        else:
            mp.step_robot(None, dt)

        for h in [h for h in self.haptics if h[0] <= now]:
            self.link.send(hap=[h[1], h[2]])
        self.haptics = [h for h in self.haptics if h[0] > now]

        q_now = self.arm.joints()
        w_now = self.arm.gripper()
        row = dict(t=now, age=min(age, 99.0), p=(p if p is not None else np.full(3, np.nan)),
                   hd=(hd if hd is not None else np.nan), tr=tr, sq=sq, emu=int(m.get("emu", 1)), track=bool(track),
                   state=STATE_CODE[self.state], target=(tgt if tgt is not None else np.full(3, np.nan)),
                   cur=mp.cur.copy(), yaw_goal=mp.yaw_goal, cur_yaw=mp.cur_yaw, q=q_now, q_cmd=mp.q_cmd.copy(),
                   grip=w_now, grip_cmd=mp.width, n_rx=n_rx)
        self.rows.append(row)
        if self.rec.on:
            self.rec.add(**row)

        if now - self.stat_t >= 1.0:
            rate = (n_rx - self.stat_n) / (now - self.stat_t)
            self.stat_t, self.stat_n = now, n_rx
            if not connected:
                line = "waiting for the Quest page (open the https address on the Quest browser)"
            elif rx_t is None:
                line = "page open, press Start on the Quest"
            else:
                st = {"idle": "stopped", "catchup": "approaching", "follow": "FOLLOWING"}[self.state]
                pos = ""
                if p is not None and self.ref is not None:
                    d = rot_z(-self.ref["h"]) @ (p - self.ref["p"]) * 100
                    pos = f" | controller from cross fwd {d[0]:+5.1f} left {d[1]:+5.1f} up {d[2]:+5.1f} cm"
                tip = self.kin.fk(q_now)[0]
                err = f" | tip-target {np.linalg.norm(tip - mp.cur) * 100:.1f} cm" if self.state != "idle" else ""
                line = (f"signal {rate:3.0f}/s | {'tracking ok' if track else '!' + why_lost} | {st}"
                        f"{'' if self.ref or a.mode != 'paper' else ' | no reference (B 1 s)'}{pos}{err}"
                        f" | gripper {(w_now if np.isfinite(w_now) else mp.width) * 1000:.0f} mm"
                        f"{' | REC' if self.rec.on else ''}")
            if line != self.last_print or self.state != "idle":
                print("[VR]", line, flush=True)
                self.last_print = line
        return row

    def finish(self):
        if self.state != "idle":
            self.arm.hold()
        if self.rec.on:
            r = self.rec.stop()
            if r:
                print(f"[기록] {r[0]} ({r[1]} 줄)")
        path = None
        if self.rows:
            os.makedirs(self.rec.dir, exist_ok=True)
            path = os.path.join(self.rec.dir, "session.npz")
            out = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
            np.savez(path, **out)
            with open(os.path.join(self.rec.dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump({"args": vars(self.a), "robot_ref": self.robot_ref.tolist(),
                           "box_lo": self.mapper.box_lo.tolist(), "box_hi": self.mapper.box_hi.tolist(),
                           "events": [(round(t - self.rows[0]["t"], 3), s) for t, s in self.events]},
                          f, ensure_ascii=False, indent=1)
        return path


# ---------------------------------------------------------------- 끝난 뒤 숫자로 보기
def _smooth_vel(x, t, k=5):
    v = np.gradient(x, t, axis=0)
    if len(v) > k:
        ker = np.ones(k) / k
        v = np.column_stack([np.convolve(v[:, i], ker, mode="same") for i in range(v.shape[1])])
    return v


def analyze(path):
    d = np.load(path, allow_pickle=True)
    try:
        with open(os.path.join(os.path.dirname(path), "session.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except OSError:
        meta = {}
    t = d["t"] - d["t"][0]
    kin = PiperKin()
    tip = np.array([kin.fk(q)[0] for q in d["q"]])
    lo = np.array(meta.get("box_lo", [-9, -9, -9]))
    hi = np.array(meta.get("box_hi", [9, 9, 9]))
    out = {"duration": float(t[-1])}
    rx = np.unique(np.round((d["t"] - d["age"])[d["age"] < 1.0], 4))
    if len(rx) > 2:
        gaps = np.diff(rx)
        out.update(rate=len(rx) / max(rx[-1] - rx[0], 1e-6), gap_max=float(gaps.max()), gaps_50ms=int(np.sum(gaps > 0.05)))
    held = d["sq"] > 0.65
    if held.any():
        out["lost_while_held"] = float(np.mean(~d["track"][held]))
    fol = d["state"] == 2
    out["follow_s"] = float(np.sum(np.diff(t, prepend=t[0])[fol]))
    # 따라간 구간 (1초 넘는 것) 마다 늦음을 재고 모은다
    segs, i = [], 0
    while i < len(fol):
        if fol[i]:
            j = i
            while j < len(fol) and fol[j]:
                j += 1
            if t[j - 1] - t[i] > 1.0:
                segs.append((i, j))
            i = j
        else:
            i += 1
    if segs:
        lags, tv, pv, errs, errs_lag, rng_t, rng_p = [], [], [], [], [], [], []
        for i, j in segs:
            tt = t[i:j]
            tg = np.clip(d["target"][i:j], lo, hi)
            tp = tip[i:j]
            vt, vp = _smooth_vel(tg, tt), _smooth_vel(tp, tt)
            dt_ = float(np.median(np.diff(tt)))
            best, best_c = 0, -1e9
            for s in range(0, int(0.6 / dt_)):
                a_, b_ = vt[:len(vt) - s] if s else vt, vp[s:]
                c = float(np.sum(a_ * b_)) / max(len(a_), 1)
                if c > best_c:
                    best, best_c = s, c
            lags.append((best * dt_, j - i))
            tv.append(np.linalg.norm(vt, axis=1))
            pv.append(np.linalg.norm(vp, axis=1))
            errs.append(np.linalg.norm(tp - tg, axis=1))
            if best:
                errs_lag.append(np.linalg.norm(tp[best:] - tg[:-best], axis=1))
            else:
                errs_lag.append(errs[-1])
            rng_t.append(tg)
            rng_p.append(tp)
        w = np.array([n for _, n in lags], float)
        out["lag_s"] = float(np.sum(np.array([l for l, _ in lags]) * w) / w.sum())
        tv, pv = np.concatenate(tv), np.concatenate(pv)
        out["speed95_target"], out["speed95_tip"] = float(np.percentile(tv, 95)), float(np.percentile(pv, 95))
        e, el = np.concatenate(errs), np.concatenate(errs_lag)
        out["err_med"], out["err95"], out["err_lag_med"] = float(np.median(e)), float(np.percentile(e, 95)), float(np.median(el))
        T, P = np.concatenate(rng_t), np.concatenate(rng_p)
        out["range_target"] = (np.percentile(T, 98, 0) - np.percentile(T, 2, 0)).tolist()
        out["range_tip"] = (np.percentile(P, 98, 0) - np.percentile(P, 2, 0)).tolist()
        out["segments"] = len(segs)
    out["events"] = meta.get("events", [])
    return out


def report(path):
    r = analyze(path)
    print(f"== {path}")
    print(f"길이 {r['duration']:.0f}초, 따라간 시간 {r['follow_s']:.0f}초")
    if "rate" in r:
        print(f"Quest 신호 초당 {r['rate']:.0f}번, 가장 긴 끊김 {r['gap_max'] * 1000:.0f} ms, 50 ms 넘는 끊김 {r['gaps_50ms']}번")
    if "lost_while_held" in r:
        print(f"쥐기 버튼을 누르고 있는 동안 추적이 안 된 때 {r['lost_while_held'] * 100:.1f}%")
    if "lag_s" in r:
        print(f"팔이 컨트롤러보다 늦은 정도 {r['lag_s'] * 1000:.0f} ms (목표가 움직인 뒤 집게 끝이 같은 만큼 움직이기까지)")
        print(f"빠르기 (95% 지점): 컨트롤러 목표 {r['speed95_target'] * 100:.0f} cm/s, 실제 집게 끝 {r['speed95_tip'] * 100:.0f} cm/s"
              f" → {r['speed95_tip'] / max(r['speed95_target'], 1e-6) * 100:.0f}%")
        rt, rp = np.array(r["range_target"]) * 100, np.array(r["range_tip"]) * 100
        print(f"움직인 폭 (앞뒤/좌우/위아래 cm): 목표 {rt[0]:.0f}/{rt[1]:.0f}/{rt[2]:.0f}, 집게 끝 {rp[0]:.0f}/{rp[1]:.0f}/{rp[2]:.0f}")
        print(f"집게 끝과 목표 차이: 가운데 {r['err_med'] * 100:.1f} cm, 95% {r['err95'] * 100:.1f} cm, "
              f"늦음을 빼면 가운데 {r['err_lag_med'] * 100:.1f} cm")
    else:
        print("1초 넘게 따라간 구간이 없음")
    return r


# ---------------------------------------------------------------- Quest 없이 시험
def fake_script(t):
    """가짜 컨트롤러 (WebXR 좌표). 기준 → 다가가기 → 위아래 → 좌우 → 추적 끊김 → 다시 → 앞뒤 → 집게 닫기 → 놓기."""
    P0 = np.array([0.0, 0.85, -0.45])                     # 종이 + (헤드셋 원점에서 앞 45 cm, 높이 85 cm)
    m = {"vis": "visible", "emu": 0, "tr": 0.0, "sq": 0.0, "a": 0, "b": 0, "q": [0.0, 0.0, 0.0, 1.0]}
    off = np.zeros(3)
    if 0.5 <= t < 1.8:
        m["b"] = 1
    if t >= 2.2:
        off[1] = 0.10                                         # 종이 위 10 cm 에서 시작
    if 2.2 <= t < 14.0:
        m["sq"] = 1.0
    if 4.5 <= t < 8.5:
        off[1] = 0.10 + 0.10 * math.sin(2 * math.pi * 0.8 * (t - 4.5))      # 위아래 ±10 cm, 최고 50 cm/s
    if 8.5 <= t < 10.3:
        off[0] = 0.12 * math.sin(2 * math.pi * 0.7 * (t - 8.5))             # 좌우 ±12 cm
    if 10.3 <= t < 12.0:
        off[0] = 0.12 * math.sin(2 * math.pi * 0.7 * (10.3 - 8.5))          # 그 자리에 멈춤
    if 10.0 <= t < 10.3:
        m["emu"] = 1                                          # 헤드셋이 컨트롤러를 놓침
    if 10.3 <= t < 10.5:
        m["sq"] = 0.0                                         # 놓았다가
    if 12.0 <= t < 13.0:
        off[0] = 0.12 * math.sin(2 * math.pi * 0.7 * (10.3 - 8.5))
        off[2] = -0.08 * math.sin(2 * math.pi * 1.0 * (t - 12.0))           # 앞뒤 ±8 cm
    if 13.0 <= t < 14.0:
        off[0] = 0.12 * math.sin(2 * math.pi * 0.7 * (10.3 - 8.5))
        m["tr"] = min(1.0, (t - 13.0) / 0.5)                  # 방아쇠 당김 → 집게 닫힘
    if t >= 14.0:
        off[0] = 0.12 * math.sin(2 * math.pi * 0.7 * (10.3 - 8.5))
        m["tr"] = 1.0
    if t > 14.6:
        return None
    m["p"] = (P0 + off + 0.0002 * np.sin(np.array([37.0, 41.0, 43.0]) * t)).round(5).tolist()   # 손 떨림 0.2 mm
    return m


async def _fake_client(port, got):
    from websockets.asyncio.client import connect
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with connect(f"wss://127.0.0.1:{port}/ws", ssl=ctx) as ws:
        async def reader():
            try:
                async for raw in ws:
                    got.append(json.loads(raw))
            except Exception:  # noqa: BLE001
                pass
        rd = asyncio.create_task(reader())
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        seq = 0
        while True:
            m = fake_script(time.monotonic() - t0)
            if m is None:
                break
            seq += 1
            m["seq"] = seq
            await ws.send(json.dumps(m))
            await asyncio.sleep(1 / 72)
        await asyncio.sleep(0.3)
        rd.cancel()


def selftest(a):
    import urllib.request
    a.robot, a.lift, a.mode = "preview", 0.10, "paper"
    a.out = os.path.join(HERE, "data", "selftest")
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(("  통과  " if ok else "  실패  ") + name)

    kin = PiperKin()
    arm = make_arm("preview")
    ctx = ensure_cert(os.path.join(HERE, "vr_cert"))
    link = QuestLink(a.port, ctx)
    if link.error:
        raise SystemExit(f"포트 {a.port} 를 못 엶: {link.error}")
    # 조종 화면이 https 로 나오는지
    c2 = ssl.create_default_context()
    c2.check_hostname, c2.verify_mode = False, ssl.CERT_NONE
    try:
        html = urllib.request.urlopen(f"https://127.0.0.1:{a.port}/", context=c2, timeout=5).read().decode()
        check("조종 화면이 https 로 나옴", "PiPER VR handle" in html and "immersive-ar" in html)
    except OSError as e:
        check(f"조종 화면이 https 로 나옴 ({e})", False)
    got = []
    th = threading.Thread(target=lambda: asyncio.run(_fake_client(a.port, got)), daemon=True)
    ctl = Controller(a, link, arm, kin, say=lambda s: print("   ", s))
    th.start()
    period = 1.0 / a.hz
    t_end = time.monotonic() + 16.5
    while th.is_alive() and time.monotonic() < t_end:
        t = time.monotonic()
        ctl.tick()
        spare = period - (time.monotonic() - t)
        if spare > 0:
            time.sleep(spare)
    for _ in range(int(0.3 * a.hz)):
        ctl.tick()
        time.sleep(period)
    path = ctl.finish()
    link.close()
    print("== 기록 숫자")
    r = report(path)
    d = np.load(path)
    t = d["t"] - d["t"][0]
    ev = [s for _, s in ctl.events]
    first = t[np.argmax(d["n_rx"] > 0)]                     # 가짜 컨트롤러 첫 신호를 0초로
    tt = t - first
    st = d["state"]
    tip = np.array([kin.fk(q)[0] for q in d["q"]])
    print("== 확인")
    check("B 1초로 기준을 잡음", any(s.startswith("reference set") for s in ev))
    reached = tt[np.argmax(st == 2)] if (st == 2).any() else 99
    check(f"쥐기 버튼 뒤 천천히 다가가서 맞춰짐 ({reached:.1f}초, 4.5초 전)", reached < 4.5)
    seg = (tt > 5.0) & (tt < 8.5)
    zr_t = np.ptp(np.clip(d["target"][seg, 2], ctl.mapper.box_lo[2], ctl.mapper.box_hi[2]))
    zr_p = np.ptp(tip[seg, 2])
    check(f"위아래 ±10 cm 를 따라감 (목표 {zr_t * 100:.0f} cm, 집게 끝 {zr_p * 100:.0f} cm)", zr_p > 0.9 * zr_t and zr_t > 0.15)
    check(f"늦음 0.1초 안 (가상 팔, {r.get('lag_s', 9) * 1000:.0f} ms)", r.get("lag_s", 9) < 0.1)
    check(f"빠르기 90% 넘음 ({r.get('speed95_tip', 0) / max(r.get('speed95_target', 1), 1e-6) * 100:.0f}%)",
          r.get("speed95_tip", 0) > 0.9 * r.get("speed95_target", 1))
    lost = (tt > 10.08) & (tt < 10.3)
    check("추적이 끊기면 멈춤", lost.any() and np.all(st[lost] == 0))
    moved = np.linalg.norm(tip[lost][-1] - tip[lost][0]) if lost.sum() > 2 else 1
    check(f"멈춘 동안 팔이 그 자리 ({moved * 1000:.1f} mm)", moved < 0.005)
    again = (tt > 12.0) & (tt < 13.0)
    check("놓았다 다시 누르면 다시 따라감", again.any() and np.mean(st[again] == 2) > 0.9)
    late = tt > 13.9
    check(f"방아쇠 당기면 집게 닫힘 ({d['grip_cmd'][late][0] * 1000:.0f} mm)", late.any() and d["grip_cmd"][late][0] < 0.005)
    after = tt > 14.1
    check("쥐기 버튼 놓으면 멈춤", after.any() and np.all(st[after] == 0))
    check("떨림 신호를 화면에 보냄", sum(1 for g in got if "hap" in g) >= 6)
    paper = d["target"][(tt > 3.0) & (tt < 4.4)]
    want = np.array([a.x0, a.y0, TABLE_Z + a.lift + 0.10])
    check(f"기준: 컨트롤러가 종이 + 위 10 cm → 집게 목표도 로봇 종이 + 위 10 cm (+lift) "
          f"({np.round((np.nanmedian(paper, 0) - want) * 1000, 1)} mm 차)",
          np.linalg.norm(np.nanmedian(paper, 0) - want) < 0.002)
    n_ok = sum(ok for _, ok in checks)
    print(f"\n결과: {'모두 통과' if n_ok == len(checks) else f'{len(checks) - n_ok}개 실패'} ({n_ok}/{len(checks)})")
    return n_ok == len(checks)


# ---------------------------------------------------------------- 실행
def main():
    a = parse()
    if a.report:
        report(a.report)
        return
    if a.selftest:
        sys.exit(0 if selftest(a) else 1)
    kin = PiperKin()
    arm = make_arm(a.robot, a.can, a.speed)
    if arm.is_real:
        print("\n[안전] 실물 팔입니다. 팔 둘레를 비우고, 전원 스위치 옆에 사람이 있어야 합니다.")
        print(f"       쥐기 버튼을 누르고 있을 때만 움직이고, 놓거나 추적이 {a.timeout:.2f}초 끊기면 멈춥니다. 속도 {a.speed}%")
        print(f"       집게 끝 최고 {a.max_speed * 100:.0f} cm/s, 책상 위 {a.min_z * 100:.0f}~{a.max_z * 100:.0f} cm 안,"
              f" 종이 위 {a.lift * 100:.0f} cm 띄워 따라감")
        input("       준비됐으면 Enter (취소 Ctrl+C) ")
    ctx = ensure_cert(os.path.join(HERE, "vr_cert"))
    link = QuestLink(a.port, ctx)
    if link.error:
        raise SystemExit(f"포트 {a.port} 를 못 엶: {link.error}  (다른 vr_handle 이 켜져 있는지)")
    ips = my_ips()
    print("[VR] Quest 브라우저에서 여세요:  " + "   ".join(f"https://{ip}:{a.port}" for ip in ips))
    print("     '연결이 비공개로 설정되어 있지 않습니다' 가 나오면 고급 → 계속. 그다음 Start (passthrough)")
    ctl = Controller(a, link, arm, kin)
    period = 1.0 / a.hz
    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic()
            ctl.tick()
            if a.run_seconds is not None and t - t0 > a.run_seconds:
                break
            spare = period - (time.monotonic() - t)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        path = ctl.finish()
        arm.hold()
        arm.close()
        link.close()
        if path:
            print(f"[끝] 기록 {path}")
            try:
                report(path)
            except Exception as e:  # noqa: BLE001
                print("[끝] 숫자 보기 실패:", e)


if __name__ == "__main__":
    main()
