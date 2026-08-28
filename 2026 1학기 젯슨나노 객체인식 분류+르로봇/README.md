# 스마트팩토리 비전 검사 시스템 (캡스톤)

> Jetson Orin Nano · 2026년 1학기 캡스톤 프로젝트 · **최종 전시회 출품작**

카메라 2대를 동시에 돌려 **박스를 분류**하고 **무게 숫자를 읽어**,
결과를 아두이노로 보내 **로봇팔을 제어**하는 스마트팩토리 시스템입니다.

```
[RealSense]  ─┐                              ┌─→ 아두이노(/dev/ttyACM0) → 로봇팔
              ├→ YOLO 병렬 추론(TensorRT FP16) ┤
[USB 웹캠]   ─┘        (FastAPI 멀티스레딩)    └─→ 웹 관제 대시보드 + Gemini 챗봇
```

---

## 이 프로젝트의 핵심 — 하드웨어 고장을 설계 변경으로 극복

원래 **CSI 카메라(IMX219)** 기반으로 GStreamer 파이프라인까지 완성해 뒀는데,
**CSI 포트가 물리적으로 고장** 났습니다.

→ Intel RealSense + 일반 USB 웹캠 조합으로 아키텍처를 갈아엎고,
FastAPI 멀티스레딩으로 두 카메라의 YOLO 추론을 지연 없이 병렬 처리하도록 재설계해 **전시회를 완주**했습니다.

> 설계 변경 과정을 증명하려고 **기존 CSI 버전 코드도 함께 보관**했습니다.

---

## 파일별 설명

### 메인 서버 (발전 순서)

| 파일 | 설명 |
|---|---|
| **`project/jetson_USB2.py`, `jetson_USB.py`** | ⭐ **최종 완성본.** 듀얼 카메라 하이브리드 제어 서버. FastAPI 멀티스레딩으로 두 카메라의 YOLO 추론(무게 숫자 인식 / 박스 분류)을 병렬 처리하고, 결과를 아두이노 시리얼로 전송 |
| `project/jetson_csi_final.py`, `jetson_main1(9).py` | **Plan A 버전.** 하드웨어 파손 이전의 CSI 카메라 기반 코드. `nvarguscamerasrc` GStreamer 파이프라인으로 하드웨어 인코딩 영상을 가져오는 고성능 지향 설계 |
| `project/jetson_main1.py` | **초기 프로토타입.** 아두이노 시리얼 통신 규격(`MOVE:RIGHT` 등)을 확립하고 TensorRT 엔진 로딩을 최초 테스트한 뼈대 코드 |

### AI 모델 최적화

| 파일 | 설명 |
|---|---|
| **`project/turnonnx.py`** | PyTorch(`.pt`) → **TensorRT 엔진(`.engine`)** 변환 스크립트. `half=True`로 **FP16 반정밀도 연산**을 활성화해 추론 속도를 크게 향상시킨 핵심 최적화 코드 |
| `project/box.pt / box.onnx / box.engine` | 박스 분류 모델 (원본 → ONNX → TensorRT) |
| `project/number.pt / number.onnx / number.engine` | 무게 숫자 인식 모델 |

### 디버깅 도구

| 파일 | 설명 |
|---|---|
| `project/cam_check.py` | 듀얼 카메라 멀티스레딩 추론 안정성 테스트. 두 카메라가 충돌 없이 YOLO 루프를 도는지, 프레임 락이 걸리지 않는지 점검 |
| `project/imx.py` | 단일 카메라 연결 확인용 초경량 스크립트. OpenCV만 열어서 `/dev/video0` 마운트 여부와 해상도를 확인 |
| `project/testgpu.py`, `test.py` | GPU 가속 / API 통신 검증 |

### 프론트엔드

| 파일 | 설명 |
|---|---|
| **`project/index.html`** | 스마트팩토리 실시간 관제 UI. 듀얼 카메라(CAM1/CAM2) 동시 스트리밍, CPU·FPS 모니터링, **우측 하단 챗봇으로 실시간 현장 데이터를 AI에게 질의**해 공정 상태를 진단받는 인터페이스 |

### 기타

| 항목 | 설명 |
|---|---|
| `project/lerobot/` | HuggingFace **lerobot** 오픈소스 (로봇팔 제어용, 직접 작성한 코드 아님) |
| `project/RIP/` | 이전 세대 코드 보관 |
| `project/*.log` | 실제 공정 운영 로그 |
| [`코드설명.md`](코드설명.md) | 파일별 상세 설명 원본 |

---

## 사용 기술

| 분류 | 내용 |
|---|---|
| 하드웨어 | Jetson Orin Nano, Intel RealSense, USB 웹캠, Arduino |
| AI | YOLO (객체 인식 + 숫자 인식), **TensorRT FP16 양자화** |
| 백엔드 | FastAPI, 멀티스레딩, 시리얼 통신(pyserial) |
| 프론트 | HTML/CSS/JS, 실시간 스트리밍, LLM 챗봇 연동 |
