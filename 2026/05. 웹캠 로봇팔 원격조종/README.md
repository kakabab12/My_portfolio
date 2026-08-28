<div align="center">

# 웹캠 하나로 로봇팔 원격조종

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logo=google&logoColor=white)
![HuggingFace](https://img.shields.io/badge/lerobot-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

**장갑도 센서도 없이, 웹캠의 2D 좌표만으로 6축 로봇팔을 조종한다**

</div>

---

사람이 **팔꿈치를 90° 굽히면 로봇도 90° 굽도록**, 관절 각도를 1:1로 전달하는 것이 목표였다.

```mermaid
flowchart LR
    A["웹캠"] --> B["MediaPipe<br/>Pose + Hand"]
    B --> C["관절각 계산<br/>(x, y 좌표만)"]
    C --> D["2단계 필터링<br/>1€ Filter + SmoothDamp"]
    D --> E["lerobot"]
    E --> F["SO-ARM-101<br/>Feetech STS3215 × 6"]
```

## 관절 매핑

| 로봇 관절 | 사람 동작 | 계산 방식 |
|---|---|---|
| `shoulder_pan` | 팔 좌우 | 손목의 어깨 대비 x 오프셋 ÷ 위팔 길이 |
| `shoulder_lift` | 팔 위아래 | `asin((어깨y − 팔꿈치y) ÷ 위팔길이)` → 실제 각도 |
| `elbow_flex` | 팔꿈치 굽힘 | `180° − 위팔·아래팔 사잇각` (폄 0° ~ 최대 150°) |
| `wrist_flex` | 손목 꺾기 | 아래팔 벡터 대비 손바닥 벡터의 **부호 있는** 각도 |
| `wrist_roll` | 손 회전 | 고정 — 2D 좌표만으로는 회전축을 추정할 수 없다 |
| `gripper` | 손 펴기/주먹 | 엄지끝 ↔ 나머지 네 손가락끝 평균거리 ÷ 손바닥 길이 |

## 기술 포인트

| 항목 | 내용 |
|---|---|
| **떨림 제거** | **1€ Filter**(속도 적응형)와 **SmoothDamp**를 2단계로 걸었다 |
| **실시간 튜닝** | 실행 중 키보드로 방향·0점·배율을 조정하고 파일로 저장한다 |
| **의존성 최소화** | 깊이 카메라나 IMU 장갑 없이 **일반 웹캠 1대**만 있으면 된다 |

## 폴더

| 폴더 | 내용 |
|---|---|
| [`MediaPipe-SO-ARM101/`](MediaPipe-SO-ARM101/) | 메인 버전 — 상세 문서 포함 |
| [`lerobot 연동판/`](lerobot%20연동판/) | lerobot 연동 버전 |

주요 파일은 `arm_tracker.py`(관절각 계산) · `teleop_so101.py`(로봇 전송) ·
`download_models.py`(MediaPipe 모델 받기) · `tuning.json`(조정값)이다.
