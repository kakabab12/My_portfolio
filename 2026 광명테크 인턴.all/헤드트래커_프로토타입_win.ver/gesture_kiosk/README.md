# gesture_kiosk — 헤드트래커 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 얼굴 랜드마크를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 + Python 3.11.5** — GPU 불필요 (2026-07-18 확정)
- 동작 체계(2026-07-18 헤드트래커 전환, head_tracker_manual.md 기반, 여러 차례 개선): **코끝=
  포인터 이동 · 입 벌리기/1.5초 응시=선택 · 양 눈 감고 0.6초=뒤로(단계별)·처음으로 ·
  입 오므리기=커서 재정렬(2026-07-21, 같은 날 2차례 교체: 볼 부풀리기→코 찡그리기→
  입 오므리기, 앞 둘은 실기 미인식돼 교체)** —
  뒤로가기 제스처는 눈썹 올리기(원안)→입 오므리기→양 눈 감기 순으로 교체됨(각각 앞머리
  가림·select와의 근육 겹침 문제 해결). 2026-07-20부터 입/눈 판정은 고정 임계값이 아니라
  **잠금 직후 캡처한 사용자 평상시 표정 기준선 + 여유값**으로 판정(정확도 개선)
- 모델: **MediaPipe FaceLandmarker(Apache-2.0) 단일** — 커서·선택·뒤로가기가 전부
  얼굴 랜드마크+블렌드셰이프 하나로 판정된다 (학습 0회 스택). CPU만으로 추론 ~30 FPS
  실측(2026-07-20, 처리 해상도 640x360 — KPI 30 FPS 달성. 구 RTMPose 포즈 방식은
  CPU 0.5 FPS로 GPU가 필수였다 — 이번 전환으로 배포가 크게 단순해짐)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — 내부망은 설치가이드.md B절
run.bat            :: 실행 — 브라우저 http://localhost:5000
```

> 모든 실행 방법(벤치마크·테스트·오프라인 번들 등)·사용법 한눈에: **[설명서.md](설명서.md)**
> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**

## 인식 동작 (2026-07-21 최신)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| (연속 상태, 이벤트 아님) | 코끝을 화면 커서로 | 안구간거리 정규화 + EMA 평활 | 포인터 이동(호버) |
| select | **입 벌리기** (즉시) | jawOpen이 평상시 기준선+여유값을 넘으면 | 선택·확인 |
| select | **1.5초 응시**(대체) | 커서가 반경 안에 머무름 — 마스크 등 대비 상시 병행 활성 | 선택·확인 |
| go_back | **양 눈 감고 0.6초 버티기** | eyeBlinkLeft/Right 둘 다 평상시 기준선+여유값 이상 유지 | 뒤로(단계별) · 처음으로(완료 화면) |
| recenter | **입 오므리기** | mouthPucker가 평상시 기준선+여유값을 넘으면 | 커서를 지금 고개 위치로 재정렬 |

- 입 벌리기·응시는 동시에 조건을 만족해도 select가 **1개만** 확정된다(공용 쿨다운).
- 뒤로가기 제스처 변천: 눈썹 올리기(원안) → 입 오므리기(앞머리 가림 문제 해결) →
  양 눈 감기(select와 같은 입 근육이라 전환 중 신호 겹침 문제 해결, 2026-07-18).
  다단계 흐름(증명서→번호→장수→확인)에서는 뒤로가기가 **한 단계씩만** 되돌아가고
  입력값은 보존된다(2026-07-20) — 완료 화면에서만 예외로 처음으로 돌아간다.
- **2026-07-20 정확도 개선**: 입벌림·눈감김 판정이 고정 임계값 대신 "잠금 직후 캡처한
  평상시 표정 기준선 + 여유값"을 쓴다 — 사람마다 평상시 블렌드셰이프 값 편차가 커서
  (실기 관찰: eyeBlink 0.1~0.6) 고정 임계 하나로는 오탐/미탐이 갈렸다.
- 잠긴 사용자(초점 맞은 얼굴 기준)의 얼굴만 인식 — **다른 사람 무시**.
- 스펙 변천: 손 검출(주먹/OK 등, 07-15 이전) → 팔 쓸기(RTMPose 포즈, 07-15~07-16) →
  **현행: 헤드트래커(얼굴 랜드마크, 07-18~)**. 팔·손을 쓸 수 없는 사용자도 동일하게 조작 가능.

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 얼굴 랜드마크(MediaPipe FaceLandmarker — 유일한 모델)
  → 사용자 잠금(person_lock: 얼굴 선명도×크기) → 커서·클릭·뒤로가기 판정(head_tracker)
  → 이벤트 전송(event_sender) + 음성 안내(announce)
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ install.bat / run.bat / make_offline_bundle.bat  # 윈도우 이식·실행 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # face_landmarker.task (빌드 타임 1회 다운로드, git 미포함)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/face_estimator.py   # 얼굴 랜드마크(MediaPipe FaceLandmarker) — 유일한 추론 모델
│   ├─ postprocess/person_lock.py    # 사용자 잠금 (얼굴 크기×선명도)
│   ├─ postprocess/head_tracker.py   # 커서 매핑 + 입벌리기/응시 클릭 + 눈감기 뒤로가기 판정
│   │                                #   (평상시 기준선 상대 판정 — _MedianCalibrator)
│   ├─ postprocess/gesture_event.py  # 확정 이벤트 공통 데이터 구조
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test
├─ tests/                   # 단위 테스트 (카메라·모델 없이 실행 가능)
├─ demo_ui/index.html       # ★ 예시 민원발급기 화면 (커서 호버 기반 + 주민등록번호 입력
│                           #   키패드 시연 — 입력값은 브라우저에만 머묾, 회사 UI 수령 시 교체)
└─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목
```

★ 표시는 **회사 키오스크 프로그램을 받으면 교체/제거되는 부분** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `run.bat` / `python scripts/run_demo.py` | 파이프라인 + 예시 UI (시연용) |
| `run.bat --headless` | 파이프라인만 — 이벤트는 `event_output` 설정대로 전송 |
| `python scripts/benchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `python -m unittest discover tests -v` | 판정·잠금 단위 테스트 |

## 회사 프로그램(UI) 연동 계약

1. 이산 이벤트(엔진→UI): `event_output.mode`(console/udp) 또는 `/data` 폴링 —
   JSON `{"class_name": "select", "conf": 0.87, "ts_sec": ..., "data": {"trigger": "mouth"}}`
2. 연속 커서(엔진→UI, 데모 UI 전용): `/data` 폴링 `status.cursor_x_ratio`/`cursor_y_ratio` —
   UDP 계약에는 실리지 않음(회사 UI의 실시간 커서 필요 여부는 협의 사항, TODO 참고)
3. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 커서가 올라간 항목을 TTS로
4. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요
5. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

## 개인정보·라이선스 주의

- 엔진은 프레임·인식값을 저장하지 않는다 (설치가이드.md F절)
- **라이선스 (2026-07-18 헤드트래커 전환 기준)**: 스택 전체가 상업 사용 가능 + 코드 공개
  (카피레프트) 의무 없음 — MediaPipe(Apache-2.0) · ONNX Runtime 등 GPU 스택 제거 ·
  pyttsx3(MPL-2.0 — 무수정 사용이라 공개 의무 없음). 추론 모델이 얼굴 랜드마크 하나뿐이라
  검토 대상이 이전(rtmlib+HaGRID 시절)보다 더 단순해졌다.
  Apache/MIT의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.

## 참고 링크

- MediaPipe FaceLandmarker (Apache-2.0): https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
