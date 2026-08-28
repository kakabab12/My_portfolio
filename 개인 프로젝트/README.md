# 개인 프로젝트

수업 과제, 개인적으로 만든 도구, 관심 있어 파본 것들을 모아둔 폴더입니다.

| 폴더 | 무엇인가 | 분류 |
|---|---|---|
| **[`MediaPipe-SO-ARM101-Teleoperation/`](MediaPipe-SO-ARM101-Teleoperation/)** | ⭐ **웹캠 하나로 6축 로봇팔 원격조종.** 센서·장갑 없이 2D 좌표만으로 사람 팔 각도를 로봇에 1:1 전달 | 개인 프로젝트 |
| **[`mediapipe 관절값으로 르로봇 제어/`](mediapipe%20관절값으로%20르로봇%20제어/)** | 위 프로젝트의 다른 버전 (SO-101 + lerobot 연동) | 개인 프로젝트 |
| **[`project고클린_alpha.ver/`](project고클린_alpha.ver/)** | ⭐ **윈도우 최적화 GUI 도구.** 청소·게임모드·핑테스트·PC정보. exe로 빌드 배포 | 개인 도구 |
| [`project고클린_초기.ver/`](project고클린_초기.ver/) | 위 도구의 초기 배치파일(.bat) 버전 | 개인 도구 |
| [`2024년 2학기 임베디드 연습/`](2024년%202학기%20임베디드%20연습/) | **임베디드 리눅스 프로젝트** — 배달 감지 카메라 / 바디캠 (보고서 + 시연 영상) | 수업 과제 |
| [`2020_express-locallibrary-tutorial/`](2020_express-locallibrary-tutorial/) | Node.js **Express + MongoDB** 도서관 웹앱 (실행화면 캡처 포함) | 수업 실습 |
| [`2024년 2학기ASP_DB연동/`](2024년%202학기ASP_DB연동/) | **ASP + DB 연동** 과제 | 수업 과제 |
| [`xampp 웹 자기소개서 간단하게/`](xampp%20웹%20자기소개서%20간단하게/) | XAMPP 기반 웹 자기소개서 | 수업 과제 |
| [`마인크래프트 fall in pickaxe/`](마인크래프트%20fall%20in%20pickaxe/) | ⚠️ **타인 오픈소스** — 학습·개조 목적으로 받아둔 코드 | 외부 코드 |

---

## 주요 프로젝트 상세

### 웹캠 하나로 로봇팔 원격조종

> 📂 `MediaPipe-SO-ARM101-Teleoperation/` · `mediapipe 관절값으로 르로봇 제어/`

**장갑도 센서도 없이 웹캠의 2D 좌표(x, y)만으로** SO-ARM-101 6축 로봇팔을 조종합니다.
사람이 팔꿈치를 90° 굽히면 **로봇도 90° 굽도록** 각도를 1:1로 전달하는 것이 목표였습니다.

```
[웹캠] → MediaPipe Pose/Hand → 관절각 계산 → 2단계 필터링 → lerobot → SO-ARM-101
        (어깨/팔꿈치/손목/손가락)   (x, y 만)   (1€ Filter + SmoothDamp)
```

| 로봇 관절 | 사람 동작 | 계산 방식 |
|---|---|---|
| `shoulder_pan` | 팔 좌우 | 손목의 어깨 대비 x 오프셋 ÷ 위팔 길이 |
| `shoulder_lift` | 팔 위아래 | `asin((어깨y − 팔꿈치y) ÷ 위팔길이)` |
| `elbow_flex` | 팔꿈치 굽힘 | `180° − 위팔·아래팔 사잇각` (폄 0° ~ 최대 150°) |
| `wrist_flex` | 손목 꺾기 | 아래팔 벡터 대비 손바닥 벡터의 **부호 있는** 각도 |
| `gripper` | 손 펴기/주먹 | 엄지끝 ↔ 나머지 손가락끝 평균거리 ÷ 손바닥 길이 |

- 떨림은 **1€ Filter + SmoothDamp 2단계 필터링**으로 제거
- 실행 중 키보드로 방향·0점·배율을 조정하고 파일로 저장 가능

**파일**: `arm_tracker.py`(관절각 계산) · `teleop_so101.py`(로봇 전송) ·
`download_models.py`(MediaPipe 모델 다운로드) · `tuning.json`(조정값)

---

### 윈도우 최적화 도구 '고클린'

> 📂 `project고클린_alpha.ver/` (GUI + exe) · `project고클린_초기.ver/` (배치파일)

시스템 관리 기능을 한 화면에 모은 개인 유틸리티입니다. **PyInstaller로 단일 exe 빌드**했습니다.

| 기능 | 내용 |
|---|---|
| **시스템 업데이트** | `winget`·`pip` 업데이트, 윈도우 업데이트/MS Store 자동 실행. `wmic`으로 **GPU 제조사(NVIDIA/AMD/Intel)를 감지해 해당 드라이버 페이지를 자동으로 열어줌** |
| **청소 & 부스트** | DNS 캐시 초기화, Temp 삭제, `dism` 이미지 정리, `netsh winsock reset`, 디스크 조각모음 |
| **게임 모드** | 무거운 백그라운드 프로그램 일괄 종료 후 탐색기 재시작으로 메모리 회수 |
| **핑 테스트** | 8.8.8.8 / 1.1.1.1 핑 결과를 **정규식으로 파싱**해 지연시간을 등급으로 평가 |
| **PC 정보** | CPU·메인보드·GPU·RAM + 내부/외부 IP 조회 |

**기술 포인트**
- **관리자 권한 자동 획득** — 권한을 확인하고 없으면 UAC 창을 띄워 스스로 재실행
- **GUI 멈춤 방지** — 모든 작업을 `threading.Thread(daemon=True)`로 분리해 "응답 없음" 방지
- 터미널 인코딩을 UTF-8(`chcp 65001`)로 바꿔 한글 깨짐 방지

---

### 임베디드 리눅스 프로젝트 — 배달 감지 카메라 / 바디캠

> 📂 `2024년 2학기 임베디드 연습/`

임베디드 리눅스 환경에서 만든 카메라 응용 프로젝트입니다.
**평상시 / 어두울 때 각각의 시연 영상**과 보고서(.hwpx), 발표자료(.pptx)가 함께 있습니다.

---

### 웹 개발 수업 과제 3종

| 폴더 | 스택 | 내용 |
|---|---|---|
| `2020_express-locallibrary-tutorial/` | Node.js, Express, MongoDB | 도서관 관리 웹앱 (MDN 튜토리얼 기반). 실행화면 캡처 4장 포함 |
| `2024년 2학기ASP_DB연동/` | ASP, DB | ASP에서 데이터베이스 연동 |
| `xampp 웹 자기소개서 간단하게/` | XAMPP (Apache+PHP+MySQL) | 웹 자기소개서 페이지 |

---

### ⚠️ 마인크래프트 fall in pickaxe (타인 오픈소스)

> 📂 `마인크래프트 fall in pickaxe/`

**직접 만든 것이 아닙니다.** [vycdev/falling-pickaxe](https://github.com/vycdev/falling-pickaxe)
(GPL-3.0) 프로젝트를 **학습·개조 목적으로** 받아둔 것입니다.
Pygame 기반 게임 구조와 YouTube 실시간 연동 방식을 참고하려고 보관했습니다.
