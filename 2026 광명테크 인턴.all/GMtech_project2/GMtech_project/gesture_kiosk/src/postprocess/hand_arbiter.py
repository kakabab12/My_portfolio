"""postprocess 모듈 — 두 손 병렬 판정 중재: **먼저 제스처를 낸 손**이 이긴다.

2026-08-04 신설(사용자 결정 — "일단 손을 잡아야지. 판정손이 로직이 느린데,
두 개 다 보고 제스처하는 게 입력되면 나머지 무시로직으로 고쳐줘").

■ 왜 바꾸는가
종전에는 hand_select가 **먼저 한 손을 고른 뒤에야** 판정이 시작됐다. 고르는
조건이 "모양이 보이는 손이 어깨너비 0.25만큼 실제로 이동"이라, 그 이동 구간이
판정 전에 통째로 버려진다 — 첫 제스처가 늦거나 아예 씹혔다(사용자 보고).
이제 게이트를 통과한 손을 **전부** 각자의 GestureFilter로 나란히 판정하고,
가장 먼저 이벤트를 낸 손을 사용자 손으로 확정한다. 나머지는 그 순간부터 무시.

■ 단일 손 추적을 유지할 근거가 사라졌다
07-31에 "무조건 한 손"으로 간 이유는 좌/우 라벨 불안정에서 오던 **배구 토스**
(포커스가 두 손 사이를 오감)였다. 그런데 한 손만 보는 로직에서도 그 현상이
그대로 났다(2026-08-04 사용자 보고) — 한 손만 보는 것이 증상을 막지 못한다.
여기서는 라벨을 되살리지 않는다: 슬롯 정체성은 **위치 연속성**으로만 잇는다
(라벨 불안정을 다시 들이지 않기 위해 — hand_select 모듈 독스트링과 같은 원칙).

■ 확정 뒤에는 종전과 같다
이긴 손을 hand_select의 추적 손으로 넘겨(adopt_tracked_hand) 앵커 고정·표시·
이음이 전부 종전 경로를 탄다. 진 손의 필터는 버린다 — 남겨 두면 그 손의 래치가
살아 있다가 다음 확정 때 엉뚱한 모양으로 시작한다.

■ 2026-08-06 10차 — 경쟁을 쓸기에서 **자세**로 교체 (judge_kind="bend")
사용자 확정("처음 판정 로직이 손이 움직여야지 잡히는데 로직을 잘못 생각한 것
같다 — 잡히고 고정"): 꺾임 판(hand_bend)의 명령은 **정지 자세**인데 입장권은
이동(쓸기 경쟁·이동 획득)이라, 손을 가만히 들어 자세를 취하는 정석 사용법이
입구에서 막히는 자기모순이 있었다. bend 모드에서는 슬롯마다 GestureFilter
대신 **자세 판정기(HandBendController)**를 쥔다 — 먼저 자세를 발화시킨 손이
잠기고 그 발화가 곧 첫 이벤트다. 정지한 손도 자세만 보이면 잡힌다.
구 쓸기 스택은 bend 모드에서 완전히 빠진다(획득 트리거 역할까지 소멸).
※쉬는 손 방어(이동 관문이 하던 소음 필터의 대체)는 휴식 존 높이 게이트가
맡는다 — _is_in_rest_zone 독스트링(가설·실측 근거).
"""
import math
import time

from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.hand_bend import HandBendController
from src.postprocess.hand_select import REENTRY_SPAN_RATIO
from src.utils.logger import get_logger

logger = get_logger("postprocess")

SLOT_MATCH_RATIO = 1.5    # 슬롯 이음 반경 = 손 폭 × N — hand_select의 연속 반경과
                          #   같은 값·같은 이유(프레임 간 이동보다 넉넉, 다른 손보다 좁게)
MAX_SLOTS = 1             # 동시에 판정할 손 수. ★2026-08-07 사용자 결정으로 2→1
                          #   되돌림("한쪽 손이 인식되면 다른쪽 손은 인식 안 되게 —
                          #   불필요한 인식 제거"): 2026-08-04에 "두 개 다" 경쟁으로
                          #   늘렸던 건 판정손 확정이 느려 첫 제스처가 씹히는 문제
                          #   때문이었는데, 10차(자세 경쟁 전환)로 판정 자체가 이미
                          #   훨씬 빨라졌고, 뒷사람이 있을 때 두 번째 슬롯이 그 손을
                          #   후보로 잡아 오탐 표면적을 넓히는 부작용이 더 커졌다.
                          #   슬롯이 하나면 구조적으로 경쟁·혼입 여지가 없다 — 먼저
                          #   잡힌 손 하나만 판정하고 나머지 관측은 애초에 후보조차
                          #   안 된다. 다시 "손이 늦게 잡힌다"는 불만이 나오면 그때
                          #   2로 복원할 것(그 경우 원인이 자세 경쟁 자체의 반응성인지
                          #   먼저 확인 — hand_bend.hold_sec 등)
SLOT_GRACE_SEC = 0.5      # 경쟁 슬롯의 소실 유예 — 이 시간 안의 검출 끊김은 슬롯
                          #   (필터째)을 유지하고 필터에 None을 먹인다.
                          #   ★2026-08-04 신설: 종전엔 한 프레임만 끊겨도 슬롯을
                          #   버려서, GestureFilter가 이미 갖고 있는 소실 기계(궤적
                          #   유예 · 래치 유예 1.5초)가 **아예 돌지 못했다** — 빠른
                          #   쓸기의 모션 블러 끊김(실측 0.35초, hand_select
                          #   EXEMPT_FRESH_SEC 근거)에서 획이 통째로 리셋된다.
                          #   07-31에 잡았던 "획 씹힘"이 다른 경로로 돌아오는 구조라
                          #   유예를 넣는다. 0.5는 blur 0.35를 덮되, 내린 손 자리에
                          #   나타난 다른 손이 필터(래치)를 승계하는 것은 짧게 끊는다
                          #   — EXEMPT_FRESH_SEC와 같은 값·같은 트레이드오프


class HandArbiter:
    """게이트를 통과한 손들을 나란히 판정하고, 먼저 이벤트를 낸 손으로 확정한다."""

    def __init__(self, config, clock=time.monotonic, judge_kind="filter"):
        # judge_kind: "filter"(종전 — 쓸기 경쟁) | "bend"(10차 — 자세 경쟁).
        # 명시 파라미터인 이유: config 내용으로 자동 판별하면 기존 쓸기 계약
        # 테스트(실물 config 사용)가 통째로 모드를 갈아탄다 — 배포 접점
        # (realtime_loop)이 hand_bend 섹션 유무를 보고 골라서 넘긴다
        self._config = config
        self._clock = clock
        self._judge_kind = judge_kind
        bend_cfg = config.get("hand_bend") or {}
        self._compete_below_max = bend_cfg.get("compete_below_shoulder_max")
        self._slots = []          # [{center_px, filter, last_sec}] — 위치로 이어지는 슬롯
        self._locked_slot = None  # 확정된 슬롯 | None(아직 경쟁 중)
        # 확정 슬롯의 소실 유예는 추적 정체성과 같은 시계를 쓴다 — hand_select가
        # release_sec 안의 재등장을 같은 손으로 잇는 동안, 중재기가 필터를 먼저
        # 버리면 이음이 무의미해진다(화면 가리킴의 수 초 소실 — 종전 경로가 버티던 것)
        self._locked_grace_sec = config["hand_select"].get("release_sec", 2.0)
        self.debug = {}           # 디버그 창 오버레이 — 확정 손(없으면 첫 슬롯)의 것

    def update(self, hand_selector, frame_width_px):
        """이번 프레임 판정 -> gesture_event | None.

        hand_selector.candidate_signals()가 준 손들을 슬롯에 잇고, 확정 전이면
        전부 판정한다. 확정 뒤에는 그 슬롯만 판정한다.
        """
        candidates = hand_selector.candidate_hands()
        shoulder_line_y_ratio = hand_selector.shoulder_line_y_ratio()
        self._sync_slots(candidates)
        if self._locked_slot is not None and self._locked_slot not in self._slots:
            self._locked_slot = None   # 확정 손이 사라졌다 — 다시 경쟁으로

        event = None
        winner = None
        for slot in self._slots:
            if self._locked_slot is not None and slot is not self._locked_slot:
                continue   # 확정 뒤 — 나머지 손은 무시(사용자 결정)
            slot_event = self._judge(slot, hand_selector, frame_width_px,
                                     shoulder_line_y_ratio)
            if slot_event is not None and event is None:
                event, winner = slot_event, slot
        self._update_debug()
        if winner is not None and self._locked_slot is not winner:
            self._lock_to(winner, hand_selector)
        return event

    def _judge(self, slot, hand_selector, frame_width_px, shoulder_line_y_ratio):
        """슬롯 하나를 그 슬롯의 판정기로 판정한다 — 자·좌표는 그 손의 것으로.

        손모양은 **여기서** 계산한다(지연 계산 — 2026-08-04 FPS 저하 수리):
        판정하지 않는 손에는 비용을 쓰지 않는다.

        유예 중(관측 없음)인 슬롯에는 None을 먹인다 — 종전 단일 손 경로가 하던
        일이다. 판정기의 소실 기계(bend: NONE_GRACE / filter: 궤적 유예·래치
        유예·복귀 삼킴 타이머)는 None 신호가 들어와야 돈다.
        """
        if self._judge_kind == "bend":
            # 자세 경쟁(10차) — 슬롯의 HandBendController에 손을 직접 먹인다
            if slot["observation"] is None:
                return slot["filter"].judge(None, hand_selector)
            center_px, hand, _, _ = slot["observation"]
            if (self._locked_slot is None
                    and self._is_in_rest_zone(center_px, hand_selector)):
                return slot["filter"].judge(None, hand_selector)
            return slot["filter"].judge(hand, hand_selector)
        if slot["observation"] is None:
            return slot["filter"].filter_signals(None, None, shoulder_line_y_ratio)
        center_px, hand, scale_ratio, _ = slot["observation"]
        signal = hand_selector.signal_for(hand, center_px)
        signal_ratio = (
            signal[0],
            (signal[1][0] / frame_width_px, signal[1][1] / frame_width_px),
            signal[2],
            signal[3],
        )
        return slot["filter"].filter_signals(signal_ratio, scale_ratio,
                                             shoulder_line_y_ratio)

    def _is_in_rest_zone(self, center_px, hand_selector):
        """잠금 경쟁 중 이 손이 **휴식 존**(어깨선 아래 깊숙이)에 있는가.

        ★2026-08-06 10차 — 이동 획득을 없애면서 그 관문이 하던 **소음 필터**의
        대체(채택 가설: 높이 게이트). 쉬는 손(배 위·늘어뜨림)도 자세로 읽히므로
        (늘어뜨린 펼친 손 = bend_down -> back 오발 위험) 잠금 전에는 어깨선
        아래 compete_below_shoulder_max(어깨너비 배수)보다 깊은 손을 관측 없음
        취급해 발화를 보류한다. 실측 근거(16:09 세션 계측 up=): 내린 손 -1.99 ·
        제스처 존 -0.45~-1.04(과도기 1건 -1.48) — 1.6이 그 사이를 가른다.
        ★잠금 뒤에는 게이트 없음 — 확정 손의 획이 아래로 내려가도 안 잘린다
        (hand_select 추적 면제와 같은 정신). 앵커(어깨) 미관측이면 게이트 없음
        (인식 우선 폴백 — 이 판의 관행). 키 삭제 시 게이트 없음.
        """
        if self._compete_below_max is None:
            return False
        frame = hand_selector.anchor_shoulder_frame()
        if frame is None or frame[1] <= 0.0:
            return False
        return (center_px[1] - frame[0]) / frame[1] > self._compete_below_max

    def _sync_slots(self, candidates):
        """관측을 슬롯에 잇는다 — 위치 최근접, 반경 밖이면 새 슬롯.

        라벨을 쓰지 않는 이유는 모듈 독스트링(배구 토스의 원인이 라벨 불안정이었다).

        소실 유예(2026-08-04): 관측이 빠진 슬롯은 즉시 버리지 않고 유예 동안
        유지한다(경쟁 SLOT_GRACE_SEC · 확정 release_sec) — 필터를 버리면 필터의
        소실 기계가 아예 못 돌아 빠른 쓸기가 획 중간에 리셋된다(07-31 획 씹힘의
        재림). 유예 중 재등장 이음은 **넓은 반경**(REENTRY_SPAN_RATIO)으로 한다:
        끊긴 동안에도 손은 움직였으므로 연속 반경(1.5)으로는 못 잇는다 —
        hand_select가 같은 이유로 같은 값을 쓴다.
        """
        now_sec = self._clock()
        unmatched = list(candidates)
        active = [slot for slot in self._slots if slot["observation"] is not None]
        graced = [slot for slot in self._slots if slot["observation"] is None]
        for slot in self._slots:
            slot["observation"] = None
        # 1차 — 직전 프레임에 보이던 슬롯만, 좁은 연속 반경으로 제 손을 잇는다.
        # 2차 — 유예 슬롯은 **남은** 관측에만, 넓은 재등장 반경으로.
        # 유예 슬롯을 1차에서 완전히 빼는 것이 본질이다(테스트가 구현 전에 잡았다):
        # 거리로는 못 가른다 — 옆 손이 유예 슬롯의 저장 위치를 **관통하는 프레임**엔
        # 그 손이 제 슬롯보다 유예 슬롯에 진짜로 더 가깝다. 그 순간 확정 필터가
        # 남의 손 궤적을 이어받아 가짜 획을 만들고, 진 손이 확정을 타고 이벤트를
        # 냈다. hand_select의 _other_tracks(2026-08-03 "쉬는 손 승계")와 같은 원칙:
        # 내 손이 안 보이는 동안 **다른 슬롯이 연속으로 따라오던** 손은 내 손의
        # 재등장일 수 없다 — 재등장은 잇지 못한 관측에서만 찾는다.
        self._assign_matches(active, unmatched, SLOT_MATCH_RATIO, now_sec)
        self._assign_matches(graced, unmatched, REENTRY_SPAN_RATIO, now_sec)
        # 유예가 끝난 슬롯만 버린다 — 새 슬롯을 받기 전에 자리를 비워, 유예
        # 슬롯이 상한(MAX_SLOTS)을 차지해 진짜 새 손을 막는 일을 줄인다
        self._slots = [
            slot for slot in self._slots
            if slot["observation"] is not None
            or now_sec - slot["last_sec"] <= self._grace_sec_for(slot)
        ]
        for entry in unmatched:
            if len(self._slots) >= MAX_SLOTS:
                break
            self._slots.append({
                "center_px": entry[0],
                "span_px": entry[3],
                "observation": entry,
                "last_sec": now_sec,
                # 슬롯 판정기 — bend 모드는 자세 상태기(10차), 종전은 쓸기 필터.
                # 키 이름(filter)은 슬롯 기계(이음·유예·확정)가 모드 무관이라 유지
                "filter": (HandBendController(self._config, clock=self._clock)
                           if self._judge_kind == "bend"
                           else GestureFilter(self._config, clock=self._clock)),
            })

    def _assign_matches(self, slots, unmatched, match_ratio, now_sec):
        """반경 안의 슬롯-관측 쌍을 **거리 오름차순**으로 잇는다.

        슬롯 순서대로 각자 최근접을 가져가게 하면 앞 슬롯이 남의 손을 가로챈다
        (테스트가 잡았다): 확정 슬롯이 리스트 앞에 있어서, 옆 손이 지나가며 그
        연속 반경에 들어온 순간 제 슬롯(더 가까움)보다 먼저 채 갔다 — 그 순간부터
        확정 필터가 남의 손 궤적을 판정한다. 가까운 쌍부터 이으면 각 손은 자기를
        따라오던 슬롯에 남는다.
        """
        pairs = []
        for slot in slots:
            if slot["observation"] is not None:
                continue
            for entry in unmatched:
                dist_px = math.dist(entry[0], slot["center_px"])
                if dist_px <= match_ratio * slot["span_px"]:
                    pairs.append((dist_px, slot, entry))
        pairs.sort(key=lambda pair: pair[0])
        for dist_px, slot, entry in pairs:
            if slot["observation"] is not None or entry not in unmatched:
                continue   # 슬롯이나 관측이 더 가까운 쌍에 이미 쓰였다
            unmatched.remove(entry)
            slot["center_px"] = entry[0]
            slot["span_px"] = entry[3]
            slot["observation"] = entry
            slot["last_sec"] = now_sec

    def _grace_sec_for(self, slot):
        """이 슬롯의 소실 유예 — 확정 손은 추적 정체성(release_sec)과 같은 시계.

        경쟁 슬롯을 더 짧게 두는 이유: 아직 누구 손인지 모르는 필터가 오래 남으면
        내린 손 자리에 나타난 **다른 손**이 그 래치를 승계한다(EXEMPT_FRESH_SEC와
        같은 트레이드오프). 확정 손은 hand_select가 정체성을 지켜 주므로 길게 가도
        승계 위험이 없다 — 짧으면 화면 가리킴의 수 초 소실에서 필터만 먼저 죽는다.
        """
        return (self._locked_grace_sec if slot is self._locked_slot
                else SLOT_GRACE_SEC)

    def _lock_to(self, slot, hand_selector):
        """이긴 손으로 확정 — 나머지 슬롯을 버리고 hand_select에 넘긴다."""
        self._locked_slot = slot
        self._slots = [slot]
        hand_selector.adopt_tracked_hand(slot["center_px"])
        logger.info("판정 손 확정: 먼저 발화한 손 (중심 %.0f,%.0f · %s 경쟁)",
                    slot["center_px"][0], slot["center_px"][1],
                    "자세" if self._judge_kind == "bend" else "쓸기")

    def _update_debug(self):
        active = self._locked_slot if self._locked_slot is not None else (
            self._slots[0] if self._slots else None)
        self.debug = {} if active is None else active["filter"].debug
