# 라즈베리파이 객체인식 → 아두이노 분류 시스템

> 2025년 2학기 학기 프로젝트 · Raspberry Pi + Arduino

카메라로 물체를 인식해 **크기(대/중/소)와 불량 여부를 판정**하고,
그 결과를 아두이노로 보내 **분류기를 제어**하는 시스템입니다.
그래픽카드가 없는 라즈베리파이에서 **YOLOv8-seg를 CPU로** 돌리는 것이 과제였습니다.

```
[카메라] → YOLOv8-seg(ONNX, CPU) → 마스크 면적(크기) + 사각형 비율(불량)
                                        → 6가지 분류 → 아두이노 SORT:x → 분류기 동작
                                        → 웹 대시보드(스트리밍 + 생산량/불량률 차트)
```

---

## 폴더가 각각 무엇인가

| 폴더 | 내용 |
|---|---|
| **[`RPI-Flask-main/`](RPI-Flask-main/)** | ⭐ **라즈베리파이 본체 코드.** Flask 서버 + YOLOv8-seg(ONNX) CPU 추론 |
| [`FastAPI_GPU버전-수정본/`](FastAPI_GPU버전-수정본/) | **GPU 버전.** 젯슨 환경으로 옮겨 FastAPI + TensorRT 가속 + Gemini 관제 챗봇 추가 |
| [`sub/`](sub/) | 학습 데이터 실험(1000장 / 1500장 세트)과 그때의 서버 코드 |
| [`Classfication_fruit.v1i.yolov8/`](Classfication_fruit.v1i.yolov8/) | Roboflow에서 받은 학습용 데이터셋 (YOLOv8 포맷) |

---

## 버전별 발전 과정

`RPI-Flask-main/` 안에 **같은 기능의 여러 세대**가 남아 있습니다.

| 파일 | 역할 | 특징 |
|---|---|---|
| `app_base.py` | 초기(Base) 버전 | 인식된 픽셀을 **3D 물리 좌표로 변환**해 아두이노에 `MOVE:x,y,z` 전송 |
| `app_ck.py` | 메인 제어 코드 | 마스크 면적(크기)과 사각형 비율(R값)로 **대/중/소 × 정상/불량 6가지 분류** → `SORT:x` 전송 |
| **`sucess.py`** | ⭐ **최종 안정판** | 불량 판별(R값) 기준 강화 + 정확한 무게중심 계산 + 타겟 고정 시 **빨간 십자선 UI** |

---

## 파일별 설명

### 하드웨어 제어

| 파일 | 설명 |
|---|---|
| `RPI-Flask-main/camera.py` | 라즈베리파이 전용 카메라 모듈. **Picamera2**로 FHD 해상도·오토포커스(AF)·화이트밸런스(AWB)를 하드웨어 레벨에서 초기화 |
| `FastAPI_GPU버전-수정본/camera.py` | **하이브리드 카메라 래퍼.** RealSense 연결을 우선 시도하고, 실패하면 **자동으로 USB 웹캠으로 폴백**하는 고가용성 구조. 백그라운드 스레드로 프레임을 읽어 서버 부하 최소화 |

### 프론트엔드

| 파일 | 설명 |
|---|---|
| `index.html` | 웹 관제 대시보드. 실시간 카메라 스트리밍, CPU/메모리/FPS 모니터링, 시스템 제어 버튼(START/STOP/CALIB), **생산량·불량률 동적 차트(Chart.js)** |
| `FastAPI_GPU버전-수정본/index.html` | 위 대시보드 + **Gemini AI 채팅 위젯** — 작업자가 실시간 현장 데이터를 기반으로 AI와 대화하며 공정을 분석 |

### 네트워크 통신

| 파일 | 설명 |
|---|---|
| `prototype.py` | 탐지 결과·시스템 로그를 외부 중앙 서버로 HTTP POST 전송하는 프로토타입 |
| `send_test.py` | 외부 서버로 가상 탐지 데이터(JSON)를 보내보는 통신 테스트 |

### 모델 관련

| 파일 | 설명 |
|---|---|
| `best.onnx` / `best.pt` | 학습된 YOLOv8-seg 모델 |
| `clean_model.py`, `onxxtst.py`, `checkt.py` | 모델 변환·검증 스크립트 |
| `dataset.yaml` | 학습 데이터 정의 |

---

## 사용 기술

| 분류 | 내용 |
|---|---|
| 하드웨어 | Raspberry Pi (Picamera2), Arduino, (GPU판) Jetson + RealSense |
| AI | YOLOv8-seg (인스턴스 세그멘테이션), ONNX Runtime, TensorRT |
| 백엔드 | Flask (RPi판) / FastAPI (GPU판), 시리얼 통신 |
| 프론트 | HTML/CSS/JS, Chart.js, 실시간 MJPEG 스트리밍 |

📖 파일별 상세 설명 원본: [`RPI-Flask-main/코드설명.md`](RPI-Flask-main/코드설명.md) ·
[`FastAPI_GPU버전-수정본/코드설명.md`](FastAPI_GPU버전-수정본/코드설명.md)
