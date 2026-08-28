# 제스처 인식 모델 — 사용 설명서

키오스크 조작용 손 제스처 인식 파이프라인. 웹캠 → MediaPipe Hand Landmarker(손) +
rtmlib RTMPose(포즈) → **"손등팔등"(손등 + 팔꿈치까지) 판정** → 안정적으로
유지되면 `next_item` 이벤트 확정. **동작 하나만 쓰는 단순화 버전**(2026-07)이다.

## 지금 상태

**완성, 실제 웹캠으로 확인됨.** `python scripts/run_demo.py` 실행 → 주먹 쥔 채로
손등·팔꿈치까지 카메라를 향하게 하고 유지 → 콘솔에 `>>> GESTURE: next_item (0.9x)`
출력. 유닛테스트 36건 통과.

## 왜 이렇게 판정하는지

**손 방향** — 손가락을 굽혔는지(주먹 모양)를 정밀하게 보는 대신, **손목→검지MCP,
손목→새끼MCP 두 벡터의 외적(cross product) 부호**로 손의 회전 방향(손등이 보이는지
손바닥이 보이는지)만 본다 — `src/inference/detector_mediapipe.py`의
`is_back_of_hand()`. 클래스 이름은 `손등팔등`.

- 주먹을 쥐면 손가락 마디가 접혀 랜드마크가 흔들리기 쉬운데, 손을 뒤집었는지(방향)
  자체는 훨씬 안정적으로 바뀌어서 더 견고하다.
- 학습이 필요 없다 — 순수 기하 계산.
- 부호는 이론적으로 유도했고(주석 참고) 실제 웹캠에서 뒤집기(`flip_orientation`)
  없이 바로 맞는 것까지 확인됐다. 다른 카메라/각도에서 반대로 나오면
  `configs/config.yaml`의 `model.mediapipe.flip_orientation: true`로 뒤집으면 된다.

**팔꿈치까지 요구** — 손만 프레임 가장자리에 걸쳐 들어온 경우(팔 전체가 안 보임)를
걸러내기 위해, 손등 방향과 별개로 **같은 쪽 팔꿈치가 포즈 추정(rtmlib)에서 신뢰도
있게 보여야만** 이벤트가 확정된다 — `src/postprocess/person_lock.py`의
`_is_elbow_visible()`. 손 판정(MediaPipe Hand Landmarker)과 팔꿈치 판정(RTMPose
포즈)은 서로 다른 모델이지만 같은 프레임 좌표계를 쓰므로 그대로 결합했다. 이
조건 때문에 `person_lock.enabled: false`로 끄면(포즈 추정 자체가 꺼짐) 이벤트가
전혀 발화하지 않는다.

**이전에 있던 select/pause_voice/cancel/go_home/sos_call/prev_item 등 다른
동작들과 손모양 6클래스(fist/palm/ok/one/like/none) 분류·사용자 손 든 높이 판정은
전부 제거했다.** 필요해지면 git 히스토리에서 복구 가능.

**남아있는 미사용 코드 — 정리 필요 여부 확인 요망:** `scripts/collect_landmarks.py`,
`scripts/train_classifier.py`, `src/inference/hand_pose_classifier.py`,
`src/inference/hand_landmark_extractor.py`는 "우리가 직접 학습시킨 손모양
분류기"를 만들려던 이전 시도의 산물인데, 지금 방식(손등 방향, 학습 불필요)에서는
더 이상 실제 파이프라인에 연결되어 있지 않다. 학습 모델 자체가 여전히 목표라면
남겨두고, 아니라면 삭제 대상.

## 폴더 구조

```
gesture_model/
├── configs/config.yaml       # 모든 튜닝값의 단일 출처 — 수정은 여기서만
├── models/weights/
│   └── hand_landmarker.task   # MediaPipe 사전학습 모델 (Apache-2.0, 약 7.8MB)
├── scripts/
│   └── run_demo.py             # [실행] 실시간 웹캠 데모 / 키오스크 연동 지점
├── src/
│   ├── capture/camera_stream.py       # 카메라 캡처 스레드
│   ├── inference/
│   │   ├── detector.py                # Detection 공통 구조 + 검출기 생성
│   │   ├── detector_mediapipe.py      # 손 랜드마크 -> 손등 방향 판정(is_back_of_hand)
│   │   ├── pose_estimator.py          # rtmlib RTMPose (person_lock용)
│   │   └── preprocessor.py            # 거울 반전
│   ├── postprocess/
│   │   ├── gesture_filter.py          # 동작 판정 FSM (next_item 하나)
│   │   └── person_lock.py             # 사용자 잠금 + 손 좌/우 귀속
│   ├── pipeline/
│   │   ├── event_sender.py            # 이벤트 출력 (console/udp)
│   │   └── realtime_loop.py           # 파이프라인 조립 (run_pipeline)
│   └── utils/                         # config_loader / logger / metrics / visualize
├── tests/                              # 카메라·모델 없이 도는 단위 테스트
└── requirements.txt
```

**직접 실행하는 파일은 1개**: `scripts/run_demo.py`. 나머지는 이 스크립트가
가져다 쓰는 내부 모듈이거나 설정 파일이다.

## 0. 설치

`gesture_model/` 폴더에서:

```
pip install -r requirements.txt
```

`models/weights/hand_landmarker.task`는 이미 받아뒀으므로 추가로 할 일 없다.

**중요 — 첫 실행 시 자동 다운로드**: `person_lock`이 켜져 있으면(기본값) `rtmlib`이
RTMPose 포즈 모델(약 40MB)을 처음 한 번 `~/.cache/rtmlib`에 자동으로 받는다.
인터넷 연결이 필요하고, 처음 실행할 때 몇 초~수십 초 더 걸린다. 이후에는 캐시를 쓴다.

---

## 1. `configs/config.yaml` — 설정

직접 실행하는 파일이 아니라, 다른 모든 모듈이 참조하는 설정 파일. 주요 항목:

| 키 | 기본값 | 의미 |
|---|---|---|
| `camera.device_id` | 0 | 웹캠 장치 번호 |
| `camera.windows_backend` | auto | 카메라가 안 열리거나 느리면 dshow/msmf로 변경 |
| `model.mediapipe.flip_orientation` | false | 손등 판정 부호가 실제와 반대면 true로 |
| `model.mediapipe.flip_handedness` | true | 왼/오른손 라벨이 실제와 반대면 반전 |
| `person_lock.enabled` | true | **끄면 이벤트가 아예 안 뜬다** — 팔꿈치 가시성 판정(포즈 추정)이 필수라서. 반드시 true로 둘 것 |
| `gestures.next_item.stable_frame_count` | 5 | 이 프레임 연속 유지해야 next_item 확정 |
| `detect.cooldown_sec` | 1.0 | 이벤트 확정 직후 재발화 방지 시간 |

---

## 2. `scripts/run_demo.py` — 실시간 데모 / 키오스크 연동

```
python scripts/run_demo.py
```

웹캠 창에 검출 박스, 사용자 잠금 얼굴 박스, 손목 위치(L/R), FPS, 최근 확정
이벤트가 표시된다. 제스처가 확정되면 콘솔에 `>>> GESTURE: next_item (0.9x)`
형태로 출력된다. `q`로 종료.

**팀원 키오스크 프레임워크와 연동하는 지점**은 파일 상단의 이 함수 하나뿐:

```python
def on_gesture_detected(label: str, confidence: float):
    print(f">>> GESTURE: {label} ({confidence:.2f})")
```

이 함수 내용을 실제 키오스크 동작(화면 전환 함수 호출, 이벤트 큐에 넣기 등)으로
바꿔 끼우면 된다. `label`은 항상 `next_item`이다.

또는 `configs/config.yaml`의 `event_output.mode`를 `udp`로 바꾸면 같은 이벤트를
JSON으로 UDP 전송한다 (`class_name`/`conf`/`ts_sec`/`hand_side`).

---

## 3. 테스트

카메라·모델 없이 도는 순수 로직 테스트:

```
python -m unittest discover tests -v
```

- `test_mediapipe_classify.py`: 손등 방향 판정 규칙(`is_back_of_hand`)
- `test_person_lock.py`: 사용자 잠금·거울 좌우 보정·손 귀속
- `test_gesture_filter.py`: next_item 판정 FSM·쿨다운

---

## 트러블슈팅

- **`카메라(device_id=0)를 열 수 없습니다`**: 다른 프로그램이 웹캠을 점유 중인지
  확인. `configs/config.yaml`의 `camera.device_id`를 0→1로 바꿔서 다른 카메라도
  시도. 열리는 데 오래 걸리거나 FPS가 낮으면 `camera.windows_backend`를
  auto→dshow 또는 msmf로 바꿔볼 것.
- **`모델 파일이 없습니다` 류 에러**: `models/weights/hand_landmarker.task`가
  있는지 확인 (약 7.8MB). 없으면
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  에서 받아 같은 경로에 저장.
- **첫 실행이 유독 느림**: `person_lock.enabled: true`면 rtmlib이 포즈 모델을
  처음 한 번 인터넷에서 받는다 (`~/.cache/rtmlib`). 정상.
- **손등을 보여도 인식이 안 되거나, 반대로 손바닥을 보였는데 인식됨**:
  `model.mediapipe.flip_orientation`을 반전.
- **손등도 보이고 팔꿈치도 보이는데 계속 인식이 안 됨**: 팔꿈치 쪽이 카메라
  프레임에서 살짝 잘려 있거나 포즈 추정 신뢰도가 낮을 수 있다 —
  `person_lock.kpt_conf_threshold`(기본 0.3)를 낮춰보거나, 몸을 조금 뒤로
  물러나 팔꿈치까지 확실히 프레임 안에 들어오게 할 것.
- **왼손/오른손 판정이 실제와 반대로 뜸**: `model.mediapipe.flip_handedness`를 반전.
- **가만히 있어도 next_item이 자꾸 잡힘(오탐)**: `detect.conf_threshold`를 올리거나
  `gestures.next_item.stable_frame_count`를 올려서 재시도.
