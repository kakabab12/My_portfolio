<div align="center">

# 🕹️ SW파일럿 — 웹 기반 로봇 이동 제어 UI

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![ROS](https://img.shields.io/badge/roslib.js-22314E?style=flat-square&logo=ros&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5_Canvas-E34F26?style=flat-square&logo=html5&logoColor=white)

**2025 SW 파일럿 과정 산출물**

</div>

---

**웹 브라우저에서 캔버스에 좌표를 찍어 로봇을 이동시키는** 관제 UI입니다.
브라우저와 ROS를 직접 연결하는 **roslib(rosbridge)** 를 사용했습니다.

```mermaid
flowchart LR
    A["🖥️ 웹 브라우저<br/>캔버스 클릭으로 목표 좌표 지정"]
    A -->|"roslib.js<br/>(WebSocket)"| B["rosbridge_server"]
    B -->|"토픽 발행"| C["🤖 ROS 로봇"]
```

## 📁 파일

| 파일 | 설명 |
|---|---|
| `web/robotics.html` | 메인 UI. 캔버스에 이동 경로를 클릭으로 지정, 좌표 목록을 좌측 패널에 표시. 마우스 위치 실시간 표시, SweetAlert2 알림 |
| `web/styles.css` | 스타일시트 |

## 🛠️ 사용 기술

| 분류 | 내용 |
|---|---|
| 로봇 통신 | **roslib.js** (rosbridge WebSocket) — 브라우저에서 ROS 토픽 직접 발행 |
| 프론트 | HTML5 Canvas, JavaScript, SweetAlert2 |

> 💡 본격적인 로봇 프로젝트는
> **[2026/02. SW파일럿 — TurtleBot3 자율주행 로봇](../../2026/02.%20SW파일럿%20-%20TurtleBot3%20자율주행%20로봇/)** 을 참고하세요.
