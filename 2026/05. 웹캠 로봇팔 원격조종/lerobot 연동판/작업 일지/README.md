# 작업 일지 — MediaPipe 팔 추적 SO-ARM-101 원격조종

웹캠 하나로 사람 팔을 추적해 SO-ARM-101 6축 로봇팔을 실시간 제어하는 프로젝트.
깊이 카메라 없이 일반 웹캠만 사용한다.

| 항목 | 내용 |
| --- | --- |
| 실행 환경 | Windows 11, Python 3.11 |
| 하드웨어 | SO-ARM-101 (Feetech STS3215 × 6), USB 웹캠, COM3 |
| 주요 라이브러리 | mediapipe 0.10.14, lerobot 0.4.4, opencv-contrib-python 5.0.0 |

---

## 날짜별 일지

| 날짜 | 파일 | 핵심 작업 | 결과 |
| --- | --- | --- | --- |
| 2026-08-21 | [2026-08-21.md](2026-08-21.md) | 초기 구현, 환경 구축, 하드웨어 연결, 매핑 정교화, GitHub 업로드 | 2D 기반 실사용 가능 |
| 2026-08-29 | [2026-08-29.md](2026-08-29.md) | 3D 전환, 조도 대응, 그리퍼 개선, 뼈 길이 제약 | 떨림 48~78% 감소 |

---

## 처리 흐름

```
[웹캠] → 조도보정(CLAHE) → MediaPipe → 3D 관절각 계산 → 뼈길이 보정
       → 1€ 필터 → SmoothDamp → lerobot → SO-ARM-101
```

---

## 진행 상황 한눈에 보기

| 항목 | 1일차 (08-21) | 2일차 (08-29) |
| --- | --- | --- |
| 좌표 방식 | 2D (x, y) | 3D world landmark |
| 모델 | pose_landmarker_lite | pose_landmarker_full |
| 프레임레이트 | 9.5 FPS | 14.9 FPS |
| 관절각 떨림 | 7.2~10.0° | 2.3~3.7° |
| 그리퍼 | 손끝 거리 (회전에 취약) | 3D 관절각 (회전 불변) |
| 조도 대응 | 없음 | CLAHE + 적응형 감마 |

---

## 해결한 주요 문제

### 1일차

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| `mp.solutions` 없음 | 0.10.35에서 legacy API 제거 | Tasks API 재작성 |
| 모델 로딩 실패 | 한글 경로를 C++ 로더가 못 엶 | 바이트 버퍼로 전달 |
| `imshow` 창 안 뜸 | lerobot이 headless OpenCV 설치 | contrib 버전 재설치 |
| **모터 전부 무응답** | **12V 잭 접촉 불량** | 잭 재체결 |
| 손목 방향 구분 못함 | 부호 없는 각도 사용 | 외적 기반 부호 각도 |
| 팔이 절반만 펴짐 | 한계 범위 + 0점 100도 어긋남 | 오프셋 −100, 범위 확대 |
| 크게 움직여야 반응 | 1€ 필터가 느린 동작을 억제 | min-cutoff 상향 |

### 2일차

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| 정면으로 뻗으면 각도 뭉개짐 | 2D 투영의 원리적 한계 | 3D + 몸통 기준 좌표계 |
| 어두우면 인식 저하 | 조도 보정 없음 | CLAHE + 적응형 감마 |
| 손 돌리면 주먹 오인식 | 손끝 거리가 투영으로 축소 | 3D 손가락 관절각 |
| 관절값 떨림 | z 추정 오차 (뼈 길이 7.7% 변동) | 뼈 길이로 깊이 역산 |

---

## 코드에서 발견한 버그

| 버그 | 증상 | 수정 |
| --- | --- | --- |
| `remap()` 내림차순 미지원 | 입력 범위가 내림차순이면 항상 0 반환 | `abs(span)`로 판정 |
| 감마 지수 뒤집힘 | 어두운 영상이 더 어두워짐 | 지수를 직접 적용 |
| `wrist_flex` 부호 없는 각도 | 위/아래 방향 구분 불가 | 외적 기반 부호 각도 |
| `__init__` 중간에 메서드 삽입 | `offsets` 속성 생성 안 됨 | 메서드를 밖으로 이동 |

---

## 참고 문헌

- Casiez, G., Roussel, N., & Vogel, D. (2012). *1€ Filter: A Simple Speed-based Low-pass
  Filter for Noisy Input in Interactive Systems.* CHI 2012.
- Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV.
- *Game Programming Gems 4* — Critically Damped Ease-In/Ease-Out Smoothing
- [MediaPipe Tasks — Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [SO-ARM-101](https://github.com/TheRobotStudio/SO-ARM100)
