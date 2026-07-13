# 제스처 인식 모델 — 사용 설명서

키오스크 조작용 손 제스처 인식 파이프라인. 웹캠 → MediaPipe Hand Landmarker로 손
21개 랜드마크 추출 → 기하학적 규칙으로 즉시 판정 → 동작 이벤트(`move_left` 등) 확정.
**학습 0회**로 바로 동작한다.

## 왜 이렇게 바뀌었는지

이전 버전은 MediaPipe Holistic으로 팔+양손 랜드마크를 뽑아 커스텀 GRU를 학습시키는
방식이었다. 팀의 실제 프로덕션 프로젝트(`GMtech_project/gesture_kiosk`, 광명테크
"제스처 인식 배리어프리 민원발급기")를 참고해 그 프로젝트가 실제로 쓰는 방식으로
전면 재작성했다:

- **학습 대신 기하 규칙.** 손가락이 펴졌는지(TIP이 PIP보다 손목에서 충분히 먼지)와
  엄지-검지 핀치 거리만으로 fist/palm/ok/one/like를 계산식으로 즉시 판정한다.
  `collect_data.py`로 사람별 데이터를 모으고 `train.py`로 학습하던 단계가 통째로 없다.
- **MediaPipe Hand Landmarker(Apache-2.0), 학습 없이 즉시 배포 가능.** 참고 프로젝트에는
  HaGRID 데이터셋으로 학습된 YOLOv10 ONNX 모델(`YOLOv10n_gestures.onnx`)도 있었지만,
  그 프로젝트 자체 문서에 "AGPL-3.0 라이선스 리스크로 상업 납품 금지"라고 명시되어 있고
  실제로 이미 폐기된 경로였다 — 그래서 이식하지 않았다.
- **사용자 잠금(person_lock) 추가.** rtmlib(RTMPose)로 사람 포즈를 추정해, 카메라
  오토포커스가 맞은(가장 선명하고 큰 얼굴) 사람에게 잠그고 그 사람 손만 인식한다.
  다른 사람이 옆에서 손을 흔들어도 반응하지 않는다.
- **동작 판정을 FSM으로 명확히 분리.** "주먹 쥐었다 펴기", "N프레임 정적 유지",
  "10초 유지" 같은 판정 로직이 `gesture_filter.py` 하나에 모여있고, 전부 config로
  튜닝 가능하며 카메라 없이 도는 단위 테스트로 검증되어 있다.

## 제스처 목록

| 동작 | 판정 방식 | 이벤트 |
|---|---|---|
| 이동 | 주먹 쥐었다 펴기 — 왼손=왼쪽, 오른손=오른쪽 | `move_left` / `move_right` |
| 선택·확인 | OK 사인(엄지+검지 맞대기) 유지 | `select` |
| 처음으로 | 양손바닥 펴서 10초 이상 유지 | `go_home` |

- 상하 이동은 없다. 줄 끝에서 다음 줄로 넘기는 랩 처리는 UI(키오스크 프레임워크) 책임.
- 좌/우 판정은 MediaPipe handedness(어느 손인지) 기준이라, **한쪽 팔이 없는 사용자도
  인식**된다(반대쪽 팔 포즈 키포인트가 없어도 동작).
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손만 인식하고 다른 사람 손은 무시한다.
- 레거시 제스처(`point`/`palm_stop`/`swipe_left`/`swipe_right`/`thumbs_up`)는
  `configs/config.yaml`의 `gestures.legacy.enabled: true`로 켜야 판정된다 (기본 꺼짐).
- 이전 버전에 있던 `call_staff`(직원호출)는 기본값에서 뺐다. 필요하면
  `gestures.two_palm.action`을 `go_home` 대신 `help_call`로 바꾸면 양손바닥 10초
  유지가 그 이벤트를 낸다 (단, go_home과 동시에 쓸 수는 없음 — 둘 중 하나).

## 폴더 구조

```
gesture_model/
├── configs/config.yaml       # 모든 튜닝값의 단일 출처 — 수정은 여기서만
├── models/weights/
│   └── hand_landmarker.task   # MediaPipe 사전학습 모델 (Apache-2.0, 약 7.8MB)
├── scripts/
│   └── run_demo.py            # [실행] 실시간 웹캠 데모 / 키오스크 연동 지점
├── src/
│   ├── capture/camera_stream.py       # 카메라 캡처 스레드
│   ├── inference/
│   │   ├── detector.py                # Detection 공통 구조 + 검출기 생성
│   │   ├── detector_mediapipe.py      # 손 랜드마크 -> 기하 규칙 판정
│   │   ├── pose_estimator.py          # rtmlib RTMPose (person_lock용)
│   │   └── preprocessor.py            # 거울 반전
│   ├── postprocess/
│   │   ├── gesture_filter.py          # 동작 판정 FSM
│   │   └── person_lock.py             # 사용자 잠금 + 손 좌/우 귀속
│   ├── pipeline/
│   │   ├── event_sender.py            # 이벤트 출력 (console/udp)
│   │   └── realtime_loop.py           # 파이프라인 조립 (run_pipeline)
│   └── utils/                         # config_loader / logger / metrics / visualize
├── tests/                              # 카메라·모델 없이 도는 단위 테스트
├── requirements.txt
└── configs/config.yaml
```

**직접 실행하는 파일은 1개**: `scripts/run_demo.py`. 나머지는 이 스크립트가
가져다 쓰는 내부 모듈이거나 설정 파일이다.

## 0. 설치

`gesture_model/` 폴더에서:

```
pip install -r requirements.txt
```

`mediapipe`, `opencv-python`, `numpy`, `pyyaml`, `onnxruntime`, `rtmlib`이 설치된다.
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
| `model.mediapipe.finger_extended_ratio` | 1.15 | 이 배율 이상 손목에서 멀면 "손가락 폄" — 손 크기가 다양한 사용자를 감안해 조절 가능 |
| `model.mediapipe.ok_pinch_ratio` | 0.35 | 엄지-검지 거리/손크기가 이 미만이면 OK로 판정 |
| `person_lock.enabled` | true | 끄면 화면 좌/우 절반 기준으로 단순 귀속 (rtmlib 없이도 동작 확인 가능) |
| `gestures.move.fist_min_frames` | 3 | 주먹이 이 프레임 연속이면 "장전" |
| `gestures.move.open_within_sec` | 0.8 | 장전 후 이 시간 안에 펴야 이동 확정 |
| `gestures.select.stable_frame_count` | 5 | OK 사인을 이 프레임 연속 유지해야 확정 |
| `gestures.two_palm.hold_sec` | 10.0 | 양손바닥을 이 시간 유지해야 go_home 확정 |
| `detect.cooldown_sec` | 1.0 | 이벤트 확정 직후 재발화 방지 시간 |

**접근성 관련**: 손가락/손 형태가 표준과 다른 사용자는 `finger_extended_ratio`나
`ok_pinch_ratio`를 조절해서 대응할 수 있다 — 재학습이 필요 없다는 게 기하 규칙
방식의 장점이다. 다만 MediaPipe의 손 검출 자체가 표준적인 손가락 5개 형태 위주로
학습되어 있어서, 손 형태가 많이 다르면 애초에 랜드마크 추출 단계에서 잘 안 잡힐 수
있다 — 이건 이 프로젝트가 못 건드리는 MediaPipe 자체의 한계다. 제스처 자체를
손가락 모양보다 팔의 이동(주먹→펴기, 좌우 위치)이나 손 전체 자세(펴짐/OK) 위주로
설계해 둔 것도 이 때문이다.

---

## 2. `scripts/run_demo.py` — 실시간 데모 / 키오스크 연동

```
python scripts/run_demo.py
```

웹캠 창에 검출 박스, 사용자 잠금 얼굴 박스, 손목 위치(L/R), 양손바닥 유지 진행 바,
FPS, 최근 확정 이벤트가 표시된다. 제스처가 확정되면 콘솔에
`>>> GESTURE: move_left (0.9x)` 형태로 출력된다. `q`로 종료.

**팀원 키오스크 프레임워크와 연동하는 지점**은 파일 상단의 이 함수 하나뿐:

```python
def on_gesture_detected(label: str, confidence: float):
    print(f">>> GESTURE: {label} ({confidence:.2f})")
```

이 함수 내용을 실제 키오스크 동작(화면 전환 함수 호출, 이벤트 큐에 넣기 등)으로
바꿔 끼우면 된다. `label`은 `move_left`/`move_right`/`select`/`go_home` 등
확정된 이벤트 이름 그대로 들어온다.

또는 `configs/config.yaml`의 `event_output.mode`를 `udp`로 바꾸면 같은 이벤트를
JSON으로 UDP 전송한다 (`class_name`/`conf`/`ts_sec`/`hand_side`).

---

## 3. 테스트

카메라·모델 없이 도는 순수 로직 테스트 (판정 규칙·잠금·FSM 검증):

```
python -m unittest discover tests -v
```

- `test_mediapipe_classify.py`: 손 랜드마크 -> 제스처 판정 규칙
- `test_person_lock.py`: 사용자 잠금·거울 좌우 보정·손 귀속
- `test_gesture_filter.py`: 이동/선택/양손유지/레거시/쿨다운 FSM

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
- **FPS가 낮음(CPU 전용)**: `person_lock.enabled: false`로 끄면 포즈 추정이
  빠지고 화면 좌/우 절반 기준으로 손을 귀속해 가벼워진다. 또는
  `model.pose_mode`를 `lightweight`로 유지(기본값).
- **가만히 있어도 이동/선택이 자꾸 잡힘(오탐)**: `detect.conf_threshold`를 올리거나
  `gestures.select.stable_frame_count` / `gestures.move.fist_min_frames`를 올려서
  재시도.
- **왼손/오른손 판정이 실제와 반대로 뜸**: `model.mediapipe.flip_handedness`를
  반전 (MediaPipe가 버전에 따라 handedness 라벨을 문서와 반대로 내는 경우가 보고돼 있음).
