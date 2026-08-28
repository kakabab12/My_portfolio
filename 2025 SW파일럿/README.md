# 2025 SW 파일럿 — 웹 기반 로봇 이동 제어 UI

> 2025 SW 파일럿 과정 산출물

**웹 브라우저에서 캔버스에 좌표를 찍어 로봇을 이동시키는** 관제 UI입니다.
브라우저와 ROS를 직접 연결하는 **roslib(rosbridge)** 를 사용했습니다.

```
[웹 브라우저] ──roslib(WebSocket)──> [rosbridge_server] ──> [ROS 로봇]
   캔버스 클릭으로 목표 좌표 지정            토픽 발행
```

---

## 파일

| 파일 | 설명 |
|---|---|
| `web/robotics.html` | 메인 UI. 캔버스에 로봇 이동 경로를 클릭으로 지정하고, 좌표 목록을 좌측 패널에 표시. 마우스 위치 실시간 표시, SweetAlert2로 사용자 알림 |
| `web/styles.css` | 스타일시트 |

## 사용 기술

| 분류 | 내용 |
|---|---|
| 로봇 통신 | **roslib.js** (rosbridge WebSocket) — 브라우저에서 ROS 토픽 직접 발행 |
| 프론트 | HTML5 Canvas, JavaScript, SweetAlert2 |

> 이 폴더는 웹 UI 부분만 담고 있습니다.
> 본격적인 로봇 프로젝트는 **[`2026 SW 파일럿/`](../2026%20SW%20파일럿/)** 을 참고하세요.
