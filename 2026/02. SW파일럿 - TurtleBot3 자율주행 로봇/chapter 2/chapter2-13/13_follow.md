# [과제 보고서] ROS2 추적 제어 및 동적 서비스 기반 라이프사이클 관리

## 1. 수행 개요 및 목표
* **과제명**: ROS2 Chapter 2 - Problem 13 (추적 제어 및 스폰/킬 서비스 활용)
* **환경**: Ubuntu 22.04 LTS, ROS2 Humble, Python 3.10
* **수행 목표**:
  1. `/spawn` 및 `/kill` 서비스를 활용하여 동적으로 로봇 엔티티를 생성하고 제거하는 라이프사이클을 구현합니다.
  2. 상대 좌표 기반 P 제어(비례 제어)를 적용하여 자연스러운 궤적으로 상대 로봇을 추적하는 알고리즘을 구축합니다.
  3. 로봇 간 충돌/만남 판정 및 `/quit` 서비스 요청 시 시스템 전체를 안전하게 종료(Clean Shutdown)하는 비동기 이벤트를 구현합니다.

---

## 2. 주요 구현 내용

### 2.1. P 제어 기반 추적 알고리즘 (`turtle_follow.py`)
`turtle1`과 `turtle2`의 `/pose` 토픽을 실시간 구독하여 Euclidean Distance와 각도 오차(Heading Error)를 산출하였습니다.
* **선속도**: $v = \min(1.5, 1.2 \times \text{distance})$
* **각속도**: $w = 4.0 \times \text{angle\_error}$ (오차를 $[-\pi, \pi]$로 정규화)

### 2.2. 비동기 서비스 호출 및 클린 셧다운
단일 스레드 실행기(Single-Threaded Executor) 환경에서 교착 상태(Deadlock)를 방지하기 위해 `call_async()` 및 `add_done_callback()`을 적용하여 `/spawn`, `/kill` 서비스를 비동기로 안전하게 실행하였습니다.

---

## 3. 실행 및 결과 검증

### 3.1. 자동 스폰 및 추적/충돌 처리
* `ros2 launch my_robot_controller turtle_follow.launch.py` 실행 시 `turtle2`가 스폰되고 `turtle1`을 향해 주행.
* 거리 threshold(0.8m) 도달 시 `turtle2` 제거 및 노드 정상 종료.

### 3.2. `/quit` 서비스 호출 테스트
* 별도 터미널에서 `ros2 service call /quit std_srvs/srv/Empty` 수행 시 두 로봇 모두 제거 후 클린 셧다운 확인.

---

## 4. 결론 및 학습 소감
이번 과제를 통해 ROS2 서비스 클라이언트/서버를 복합적으로 활용하여 시뮬레이션 내 로봇 라이프사이클을 동적으로 제어하는 방법을 습득했습니다. 비동기 호출 패턴을 통해 데드락 없이 자원을 안전하게 해제하는 백그라운드 로직의 중요성을 재확인했습니다.