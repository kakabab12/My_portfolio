# TODO — 작업 분해 및 진행 상황 (라즈베리파이5판)

작성: 2026-07-22 (win.ver에서 이식). win.ver의 changelog 전체 이력은
`헤드트래커_프로토타입_win.ver/gesture_kiosk/docs/TODO.md` 참고 — 여기는 이 판
자체의 진행 상황만 기록한다.

## ✅ 완료 (2026-07-22 — 라즈베리파이5 이식)

- [x] **win.ver → rpi.ver 이식**: 판정 로직(postprocess/*, inference/face_estimator.py,
      preprocessor.py)·연동 계층(announce·event_sender·demo_server)·데모 UI·테스트는
      플랫폼 무관이라 그대로 복사. 카메라 캡처(`src/capture/camera_stream.py`)와
      프로세스 우선순위(`src/pipeline/realtime_loop.py`)만 새로 작성.
- [x] **카메라 백엔드 2종**: v4l2(USB 웹캠, 기본) + picamera2(공식 카메라 모듈 —
      libcamera 하드웨어 ISP로 리사이즈·BGR 변환을 대신해 CPU를 아낀다). config의
      `camera.backend`로 선택.
- [x] **systemd 배포 유닛**(`deploy/gesture_kiosk.service`) 신설 — 리눅스는 일반
      사용자가 프로세스 우선순위를 못 올려(윈도우와 다름) `Nice=-5`로 확실히 보장.
- [x] **설치·실행 스크립트**(`install.sh`/`run.sh`) — win.ver의 install.bat/run.bat과
      동일한 단계를 apt/bash로.

## 🔴 실기 검증 필요 (이 세션은 윈도우 PC라 라즈베리파이5 실기 테스트 불가)

- [ ] **`configs/config.yaml`의 추정값 검증** — `proc_width_px/proc_height_px`(480x270)·
      `num_faces`(2)·`infer_scale_ratio`(1.0)를 `scripts/benchmark.py` 실측 기준으로
      재조정. 설치가이드.md "G. 성능 튜닝" 절 참고.
- [x] ~~**mediapipe·opencv-python aarch64 휠 존재 확인**~~ — 2026-07-22 사용자 실기에서
      `mediapipe==0.10.35`(win.ver와 동일 버전) 설치가 휠 문제로 실패해 확인됨. PyPI
      조회 결과 구글이 0.10.18 이후로 리눅스 aarch64 휠 배포를 중단한 것으로 확인 —
      `mediapipe==0.10.18`(aarch64+cp311 휠 존재)로 낮추고, 그에 맞춰 `numpy<2`로
      조정, 자체 `opencv-python` 핀은 제거(mediapipe의 opencv-contrib-python
      의존성이 자동 해결됨). `requirements.txt` 상단 주석 참고.
- [ ] **picamera2 백엔드 실기 테스트** — `src/capture/camera_stream.py`의
      `_open_picamera2`/`_capture_loop_picamera2`가 실제 카메라 모듈에서 정상 동작하는지
      (BGR888 포맷 요청이 의도대로 나오는지 포함).
- [ ] **pyttsx3 + espeak-ng 리눅스 동작 확인** — 자동 드라이버 인식이 안 되면
      `announcer.py`의 `pyttsx3.init()`에 `driverName='espeak'` 명시가 필요할 수 있음.
- [ ] **`os.nice()` 우선순위 조정 동작 확인** — 일반 사용자 권한으로 얼마나 실패하는지,
      systemd 유닛(Nice=-5) 경로로 실제 체감 성능 차이가 있는지.
- [ ] **열/전력 스로틀링 확인** — `vcgencmd get_throttled`, 공식 액티브 쿨러 유무별
      장시간 구동 시 FPS 저하 여부.

## 🔴 회사 확인 필요 — win.ver와 공통 (이 판에서 별도로 결정할 사항 아님)

- [ ] **회사 UI의 실시간 커서 필요 여부** — UDP로 연속 좌표를 전달할지 별도 협의 필요.
- [ ] **회사 키오스크 프로그램(UI) 파일 수령** — 수령 후 demo_ui 교체 작업 시작.
- [ ] **이벤트 전달 규격 합의** — udp JSON 예시 구현 상태. 소켓/시리얼 등 확정 시
      `event_sender.py`에 Sender 추가.
- [ ] **통합 범위 확정** — UI 통합까지인지 이벤트 출력까지인지.
- [ ] **KPI 측정 기준 합의** — 정확도 85% 산식 / 30 FPS(엔드투엔드 vs 추론 단독) —
      라즈베리파이5는 이 판 자체의 실측이 아직 없어 win.ver 수치(CPU 단독 23~30 FPS)로
      가늠할 수 없음.
- [ ] **라이선스 문서 동봉** — MediaPipe(Apache-2.0) 배포물 동봉 절차, win.ver와 동일.
- [ ] **코드 저장 위치·백업 정책** — 이 폴더를 별도 git 저장소로 둘지, win.ver
      저장소와 통합할지 미정(요청받지 않아 git init 하지 않음).

## 🟡 이 판에서 아직 안 만든 것 (요청 범위 밖이라 스킵)

- [ ] 오프라인(내부망) 설치 번들 스크립트 — win.ver의 `make_offline_bundle.bat` 상당.
      필요해지면 `pip download -d wheelhouse/` 방식으로 추가.
- [ ] docs/CODE_SETUP.md·VSCODE_RUN.md의 라즈베리파이판 — win.ver 버전이 윈도우
      IDE 전제라 그대로 못 씀. 필요해지면 별도 작성.

## 메모

이 판의 `head_tracker.*` 제스처 임계값은 win.ver 2026-07-21(입 오므리기로 재정렬
제스처 확정된 시점) 기준을 그대로 가져왔다. win.ver에서 이후 제스처·임계값이
바뀌면 이 판에도 반영할지 판단 필요(브랜치 간 자동 동기화 없음 — CLAUDE.md
"브랜치 간 merge 금지" 원칙에 따라 수동으로 `git checkout` 후 재적용하는 방식과
동일하게, 필요한 파일만 골라 옮길 것).
