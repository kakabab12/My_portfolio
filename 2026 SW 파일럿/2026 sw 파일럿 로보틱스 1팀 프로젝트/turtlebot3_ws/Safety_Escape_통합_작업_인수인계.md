# TurtleBot3 Nav2 Safety Stop / Scan / Escape 통합 작업 인수인계

작성일: 2026-08-19

## 0. 이 문서의 목적

이 문서는 현재 TurtleBot3 Burger의 저장 지도 기반 Nav2 Waypoint 순찰 코드에 **Safety Stop → Scan Motion → Escape → 기존 Waypoint 복귀** 기능을 추가하기 위해, 지금까지 정리한 설계 방향과 현재 코드 상태를 Codex에 인수인계하기 위한 문서다.

다음 작업에서 이 문서를 먼저 읽고 기존 코드를 확인한 뒤 수정한다.

> 핵심 원칙: **Nav2 전체를 종료하지 않는다. 현재 Navigation Goal/Task만 Cancel하고, Safety Stop/Scan/Escape가 로봇 제어권을 독점한 뒤, Escape 완료 후 중단되었던 기존 Waypoint를 Nav2에 새 Goal로 다시 전송한다.**

---

# 1. 현재 시스템 구성

- ROS 2: **Humble**
- 로봇: **TurtleBot3 Burger**
- 컴퓨팅 보드: **Jetson Orin Nano**
- 라이다: **LDS-03**
- 워크스페이스: `~/turtlebot3_ws`
- 저장 지도:
  - `~/turtlebot3_ws/maps/square_2m.yaml`
  - `~/turtlebot3_ws/maps/square_2m.pgm`
- 위치추정: **AMCL**
- Navigation: **Nav2**
- 시각화: **RViz2**
- Waypoint 제어: `nav2_simple_commander.BasicNavigator`

새 터미널에서는 기본적으로 다음 환경을 로드한다.

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
```

TurtleBot3 bringup 시:

```bash
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03
```

---

# 2. 현재 실제 실행 구조

현재 테스트는 크게 3개 프로세스 그룹으로 나뉘어 있다.

## Terminal 1 — TurtleBot3 Bringup

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-03

ros2 launch turtlebot3_bringup robot.launch.py
```

역할:

- OpenCR / 모터
- odometry
- TF
- LDS-03 `/scan`
- TurtleBot3 하드웨어 bringup

## Terminal 2 — Nav2 + AMCL + Map + RViz2

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:=$HOME/turtlebot3_ws/maps/square_2m.yaml
```

역할:

- Map Server
- AMCL
- Nav2 Planner / Controller / BT Navigator
- Global / Local Costmap
- RViz2

## Terminal 3 — Waypoint Patrol

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash

ros2 launch turtlebot3_waypoint_patrol patrol.launch.py
```

현재는 한 바퀴 순찰 후 종료하도록 단순하게 구성되어 있다.

최종 시연 단계에서는 위 구성과 Mission Manager를 **하나의 통합 launch 파일**로 묶는 것이 목표다. 하지만 개발/디버깅 단계에서는 터미널을 분리하여 각 기능을 개별 검증한다.

---

# 3. 현재 Waypoint 코드의 최신 상태

실제 최신 파일:

```text
src/turtlebot3_waypoint_patrol/
└── turtlebot3_waypoint_patrol/
    └── patrol_node.py
```

현재 `patrol_node.py`는 다음 Waypoint를 사용한다.

```python
WAYPOINTS = (
    ('A', 0.095, -0.053, -0.175),
    ('B', 1.217, -0.396, -0.614),
    ('C', 1.225, -0.922, -1.799),
    ('D', 0.832, -1.504, 2.962),
    ('E', -0.129, -1.116, 2.081),
    ('A', 0.095, -0.053, -0.175),
)
```

현재 경로:

```text
A → B → C → D → E → A → 종료
```

**주의:** 기존의 2026-08-17 인수인계 Markdown에는 옛날 `A → B → C → D → A`가 남아 있으므로, Waypoint 관련해서는 현재 `patrol_node.py`를 최신 기준으로 사용한다.

현재 핵심 호출은 다음과 같다.

```python
navigator.followWaypoints(poses)
```

즉 A/B/C/D/E/A를 각각 별도의 NavigateToPose Goal로 보내는 것이 아니라, **Waypoint 리스트 전체를 FollowWaypoints Task 하나로 Nav2에 넘기고 있다.**

이 구조는 현재 단순 순찰에는 적합하지만, Safety 기능에서 다음 동작을 만들기에는 불편하다.

```text
B로 이동 중
→ 장애물 검출
→ 현재 B 이동 취소
→ Escape
→ 다시 B를 Goal로 전송
```

따라서 Safety 기능 통합 단계에서는 Waypoint 실행 구조를 변경할 예정이다.

---

# 4. Safety 기능의 최종 동작 원칙

최종적으로 사용할 구조는 다음과 같다.

```text
[NAVIGATING]
Nav2가 현재 Waypoint로 정상 주행
        │
        │ 예상하지 못한 전방 장애물 검출
        ▼
[SAFETY_STOP]
즉시 정지
        │
        ▼
[CANCEL_NAVIGATION]
현재 Nav2 Navigation Goal/Task Cancel
        │
        ▼
[SCAN_MOTION]
Safety/Recovery 로직이 독점 제어
        │
        ▼
[ESCAPE]
장애물 영역에서 안전하게 탈출
        │
        ▼
[RESEND_GOAL]
중단 당시 기억해둔 Waypoint를
Nav2에 새로운 Goal로 다시 전송
        │
        ▼
[NAVIGATING]
현재 위치 기준으로 Nav2가
새 경로를 계산해 정상 주행 재개
```

가장 중요한 원칙은 다음 두 가지다.

1. **Nav2 Stack 자체를 종료하지 않는다.**
2. **현재 Navigation Task만 Cancel하고, Escape 완료 후 동일 목적지를 다시 전송한다.**

즉 아래 노드들은 그대로 살아 있어야 한다.

```text
map_server
amcl
planner_server
controller_server
bt_navigator
behavior_server
local_costmap
global_costmap
```

Safety 동작 중에도 AMCL, TF, `/scan`, Costmap 등은 계속 유지한다.

---

# 5. "우선순위"가 아니라 "제어권 전환"으로 설계

이 시스템을 다음처럼 생각하지 않는다.

```text
1순위 Safety
2순위 Nav2
```

Nav2와 Recovery가 동시에 `/cmd_vel` 경쟁을 하는 Priority 구조를 처음부터 만들지 않는다.

대신 **Mission Manager가 현재 로봇을 누가 제어할지를 명확하게 전환**한다.

## 정상 주행

```text
Mission Manager
      ↓
NAVIGATION MODE
      ↓
Nav2만 로봇 제어
```

## 장애물 검출

```text
Mission Manager
      ↓
Nav2 Task Cancel
      ↓
SAFETY / ESCAPE MODE
      ↓
Safety/Recovery만 로봇 제어
```

## Escape 완료

```text
Safety/Recovery 종료
      ↓
현재 Waypoint 재전송
      ↓
NAVIGATION MODE
      ↓
Nav2가 다시 로봇 제어
```

**Nav2와 Safety/Recovery가 동시에 서로 다른 `/cmd_vel`을 보내지 않도록 해야 한다.**

---

# 6. 기존 `followWaypoints()` 구조를 어떻게 바꿀 것인가

현재:

```python
navigator.followWaypoints(poses)
```

권장 변경 방향:

```text
Waypoint를 하나씩 관리하는 Mission Manager
```

개념적으로:

```python
current_waypoint_index = 0

while current_waypoint_index < len(waypoints):
    current_goal = waypoints[current_waypoint_index]

    navigator.goToPose(current_goal)

    # navigation 상태 감시
    # safety trigger 발생 시 cancel -> escape -> current_goal 재전송

    # 정상 도착했을 때만 index 증가
    current_waypoint_index += 1
```

예를 들어 B로 이동 중 Safety 이벤트가 발생하면:

```text
current_waypoint = B
        ↓
navigator.cancelTask()
        ↓
Safety Stop
        ↓
Scan Motion
        ↓
Escape
        ↓
navigator.goToPose(B)
        ↓
B 정상 도착
        ↓
current_waypoint_index += 1
        ↓
C 이동
```

즉 **Goal Cancel은 Waypoint 완료로 취급하지 않는다.**

Waypoint index는 해당 목적지에 실제로 정상 도착했을 때만 증가한다.

`followWaypoints()`를 유지하면서 현재 index 이후 리스트를 재전송하는 방법도 가능하지만, 지금 프로젝트 규모와 디버깅 난이도를 고려하면 **Waypoint별 `goToPose()` State Machine 방식이 우선 권장안**이다.

---

# 7. 권장 노드 구조 — `mission_manager.py`

기존 `patrol_node.py` 기능을 확장하여 최종적으로 다음과 같은 Mission Manager 역할을 수행하도록 만드는 것을 권장한다.

```text
mission_manager.py
│
├── Waypoint Mission 관리
│   ├── current_waypoint_index
│   ├── current_goal
│   ├── goToPose()
│   ├── cancelTask()
│   └── Goal 재전송
│
├── Safety Monitor
│   ├── /scan 구독
│   ├── /map 활용
│   ├── TF 조회
│   └── Unexpected Obstacle 판정
│
├── Safety Stop
│
├── Scan Motion
│
├── Escape
│
└── Nav2 Resume
```

프로젝트 규모상 처음부터 여러 개의 노드를 Topic/Service로 복잡하게 연결하기보다, **Waypoint와 Safety/Recovery를 하나의 Mission Manager 안에서 상태 머신으로 통합**하는 것이 구현/디버깅에 유리하다.

기존 패키지 `turtlebot3_waypoint_patrol` 안에서 `mission_manager.py`를 추가하거나 `patrol_node.py`를 단계적으로 리팩터링하는 방향이 적절하다.

---

# 8. 권장 State Machine

초기 상태 후보:

```text
NAVIGATING
SAFETY_STOP
CANCEL_NAV
SCAN_LEFT
SCAN_RIGHT
RETURN_HEADING
ESCAPE
RESEND_GOAL
MISSION_COMPLETE
```

개념적인 상태 전이:

```text
NAVIGATING
    │
    │ Safety Trigger
    ▼
SAFETY_STOP
    │
    ▼
CANCEL_NAV
    │
    │ Nav2 Task Cancel 확인
    ▼
SCAN_LEFT
    │
    ▼
SCAN_RIGHT
    │
    ▼
RETURN_HEADING
    │
    ▼
ESCAPE
    │
    │ Escape 완료
    ▼
RESEND_GOAL
    │
    ▼
NAVIGATING
```

정상 도착인 경우:

```text
NAVIGATING
    │
    │ Goal SUCCEEDED
    ▼
next waypoint
    │
    └─ 마지막이면 MISSION_COMPLETE
```

---

# 9. Safety Stop은 단순 거리 임계값만 사용하지 않는다

처음에는 다음과 같은 단순 구조를 생각했다.

```text
/scan에서 전방 일정 거리 이내 장애물
→ 무조건 Safety Stop
```

하지만 이 방식은 정상적인 벽, 코너, 좁은 구간 근처에서 오검출할 수 있다.

예:

```text
정상 Nav2 코너 주행
→ 전방 LiDAR에 가까운 벽 검출
→ 거리 기준만 사용하면 Safety Stop 오작동 가능
```

따라서 현재 방향은 **"현재 LiDAR에는 존재하지만 저장된 Static Map에서는 원래 Free Space였던 전방 장애물"**을 Unexpected Obstacle 후보로 판단하는 것이다.

---

# 10. Safety Trigger에 사용할 정보

권장 입력:

```text
1. /scan
   → 현재 LDS-03이 실제로 관측하는 장애물

2. /map
   → 저장된 square_2m Static Global Map

3. TF / AMCL localization
   → 현재 scan point를 map 좌표로 변환하기 위한 로봇 위치

4. 현재 Mission 상태
   → NAVIGATING 중인지 여부
```

굳이 Nav2가 "이 물체는 Dynamic Obstacle이다"라는 별도 판정 결과를 제공하기를 기다릴 필요는 없다.

Safety Monitor가 `/scan + /map + TF`를 이용하여 직접 비교할 수 있다.

개념:

```text
현재 /scan에서 가까운 장애물 발견
              │
              ▼
scan endpoint를 map frame으로 변환
              │
              ▼
해당 위치의 Static Map 확인
        ┌─────┴─────┐
        │           │
    OCCUPIED       FREE
        │           │
기존 지도 장애물   신규/예상 밖 장애물 후보
        │           │
   Safety 제외      추가 조건 확인
```

---

# 11. 권장 Safety Trigger 조건

초기 구현은 한 조건이 아니라 여러 조건을 AND로 묶는다.

```text
현재 NAVIGATING 상태
        AND
로봇 전방 sector에서 가까운 장애물 검출
        AND
단일 noise point가 아니라 여러 LaserScan point가 존재
        AND
해당 scan point 위치가 Static Map에서는 Free Space
        AND
한 프레임이 아니라 여러 scan frame에서 연속 검출
        ↓
SAFETY TRIGGER
```

중요:

- 전방 각도 범위는 실제 테스트 후 튜닝한다.
- Safety 거리 threshold도 실제 주행속도/정지거리 확인 후 튜닝한다.
- 최소 Laser point 개수도 실제 LDS-03 scan을 보고 튜닝한다.
- 연속 검출 frame 수 역시 테스트 후 결정한다.

현재 단계에서는 특정 수치를 코드에 고정해 버리지 말고 **ROS Parameter로 노출하는 것을 권장**한다.

예시 파라미터 이름:

```yaml
safety_monitor:
  front_angle_deg: ...
  stop_distance: ...
  min_obstacle_points: ...
  consecutive_frames: ...
  static_map_tolerance: ...
```

---

# 12. Static Map 비교 시 반드시 tolerance를 둔다

현재 `square_2m` 지도와 Nav2 costmap resolution은 **0.05 m/cell**이다.

AMCL 위치추정 오차, scan 오차, 지도 discretization 때문에 scan point가 원래 벽의 정확한 cell에 떨어지지 않을 수 있다.

따라서 다음 방식은 금지한다.

```text
scan endpoint가 가리킨 딱 1개 map cell만 확인
FREE면 무조건 신규 장애물 판정
```

대신 scan endpoint 주변 일정 범위에 Static Occupied Cell이 존재하는지 검사하는 tolerance를 둔다.

개념:

```text
scan point 주변 N cells 확인
        │
        ├─ 주변에 기존 occupied cell 존재
        │      → 원래 지도상의 벽/설비 가능성
        │
        └─ 주변도 모두 free
               → 예상하지 못한 장애물 가능성 증가
```

`tolerance` 크기는 실제 AMCL 및 scan 정합 상태를 보면서 튜닝한다.

---

# 13. Nav2 Costmap의 현재 기본 설정

현재 `ROS_DISTRO=humble`, `TURTLEBOT3_MODEL=burger`이므로

```text
src/turtlebot3_navigation2/param/humble/burger.yaml
```

이 사용된다.

현재 확인된 주요 값:

```yaml
local_costmap:
  width: 3
  height: 3
  resolution: 0.05
  robot_radius: 0.1

inflation_layer:
  inflation_radius: 0.25

scan:
  topic: /scan
  obstacle_max_range: 2.5
  raytrace_max_range: 3.0
```

Global Costmap도:

```yaml
resolution: 0.05
robot_radius: 0.1
inflation_radius: 0.25
obstacle_max_range: 2.5
raytrace_max_range: 3.0
```

현재 값들은 아직 프로젝트용으로 본격 튜닝한 값이 아니라 **기본 설정에 가까운 상태**다.

당장 Safety 기능을 만들기 전에 무조건 수정할 필요는 없다.

우선 현재 설정으로 실제 주행/장애물 인식/RViz costmap을 확인하고, 문제가 확인된 항목만 단계적으로 수정한다.

Safety Trigger 자체는 Nav2 Costmap update만 기다리는 방식보다 `/scan`을 직접 감시하는 별도의 Safety Monitor 로직으로 만든다.

Nav2 Costmap은 기존 역할대로 planning/controller의 일반 장애물 회피 및 재계획에 사용한다.

---

# 14. 저장 지도 / AMCL / Local Costmap 역할 구분

혼동하지 말아야 할 점:

## Static Global Map

SLAM Toolbox로 미리 작성하고 저장한 지도다.

Nav2 주행 중 이 지도가 한 바퀴 돌 때마다 다시 SLAM 되어 보정되는 것은 아니다.

## AMCL

저장된 지도 + `/scan` + odometry 등을 바탕으로 현재 TurtleBot이 `map` 좌표계의 어디에 있는지 추정한다.

정상 주행을 하면서 localization이 안정될 수 있다.

## Local Costmap

로봇 주변의 현재 장애물 상황을 Nav2가 판단하기 위한 rolling costmap이다.

Local Costmap 자체가 AMCL처럼 위치를 추정하는 것은 아니다.

---

# 15. 시연 전 Localization 안정화 방향

시연에서는 처음부터 바로 Safety 이벤트를 만들기보다 다음 순서를 권장한다.

```text
로봇 초기 위치 설정
        ↓
Nav2 / AMCL 활성화
        ↓
Waypoint 정상 순찰
        ↓
RViz에서 localization / scan-map 정합 확인
        ↓
필요 시 추가 정상 순찰
        ↓
AMCL 위치추정이 안정적이라고 판단
        ↓
Safety 기능 시연
```

"몇 바퀴를 반드시 돌아야 한다"가 핵심이 아니다.

확인해야 할 것은 다음이다.

- RViz의 로봇 위치가 실제 위치와 유사한가
- `/scan`이 저장된 지도 벽/구조물과 대체로 잘 겹치는가
- AMCL particle cloud가 비정상적으로 넓게 퍼지거나 튀지 않는가
- 회전/직선 주행 후 pose가 크게 점프하지 않는가
- 정상 Waypoint 순찰이 안정적으로 가능한가

Safety Monitor가 Static Map과 Live Scan을 비교하므로, **AMCL localization이 불안정하면 원래 지도상의 벽을 신규 장애물로 오판할 가능성**이 있다.

따라서 Safety 기능 활성화 전에 localization 상태를 확인하는 것은 오작동 감소에 중요하다.

초기 버전에서는 자동 localization quality 판정을 복잡하게 구현하기보다, 시연자가 정상 순찰 및 RViz 정합 상태를 확인한 뒤 Safety 시연을 진행해도 충분하다.

---

# 16. Safety 이벤트 발생 후 정확한 제어 순서

구현 시 순서를 명확하게 유지한다.

```text
1. Unexpected Obstacle 검출

2. Safety Stop 시작
   - 로봇 정지 명령

3. 현재 Nav2 Navigation Task Cancel 요청

4. Nav2 Cancel 상태 확인
   - Nav2가 더 이상 정상 주행 명령을 소유하지 않도록 함

5. Safety/Scan/Escape가 제어권 획득

6. Scan Motion 수행

7. Escape 수행

8. Escape 완료 후 반드시 정지 상태 확인

9. 중단되었던 current_waypoint를 다시 Nav2 Goal로 전송

10. Nav2가 현재 위치 기준으로 새 경로 계산

11. NAVIGATING 상태 복귀
```

**주의:** Nav2 Task Cancel 확인 전에 Recovery가 무작정 `/cmd_vel`을 발행하는 구조는 피한다.

---

# 17. Scan Motion의 목적

TurtleBot의 LDS-03은 360° 2D LiDAR이므로, 기술적으로는 로봇이 제자리 회전을 해야만 좌/우 공간을 볼 수 있는 것은 아니다.

Safety Trigger 시점의 `/scan`만으로도 주변 방향의 공간 정보를 상당 부분 알 수 있다.

그럼에도 Scan Motion을 사용하는 이유는 시연에서 다음 판단 과정을 **로봇의 물리적 행동으로 시각화**하기 위해서다.

```text
정지
→ 한쪽 방향 확인 모션
→ 반대쪽 방향 확인 모션
→ 기준 heading 복귀
→ Escape
```

따라서 설명할 때는:

> LiDAR가 회전해야만 공간을 알 수 있어서 돌린다.

라고 설명하지 않는다.

대신:

> LiDAR로 주변 공간을 판단하고, 장애물 회피 판단 과정을 시각적으로 표현하기 위해 Scan Motion을 수행한다.

라고 설명한다.

---

# 18. Scan / Escape에서 회전 제어 시 주의

제자리 회전을 단순히 다음처럼 시간 기반으로만 구현하는 것은 오차가 누적될 수 있다.

```python
angular_z = fixed_value
sleep(fixed_time)
```

가능하면 다음 중 하나를 사용한다.

1. odometry/TF yaw feedback 기반 회전
2. Nav2 behavior server의 Spin/BackUp 기능 검토
3. 직접 custom recovery를 구현하되 목표 각도/거리 feedback을 확인

현재 Humble Nav2 설정에는 기본 recovery plugin이 존재한다.

```yaml
recovery_plugins: ["spin", "backup", "wait"]
```

다만 기존 Safety 시나리오와 제어권 전환을 명확히 하기 위해, 실제 구현 방식은 현재 코드와 테스트 결과를 보고 결정한다.

---

# 19. LDS-03 / scan 현재 구성

현재 프로젝트에서 사용하는 라이다 기준:

```text
LDS_MODEL=LDS-03
```

현재 프로젝트에 포함된 driver/config 기준 핵심:

```text
scan topic: /scan
frame: base_scan
```

Safety Monitor는 `/scan`을 직접 구독한다.

전방은 로봇 기준 +X 방향으로 정의한다.

TurtleBot3 ROS 좌표계 개념:

```text
          +X = Front
              ↑
              │
      +Y ← TurtleBot → -Y
              │
              ↓
             -X
```

Nav2 정상 주행에서도 기본적으로 로봇의 앞(+X)을 진행 방향으로 맞추어 이동하는 구조로 본다.

---

# 20. 오작동 방지를 위한 추가 고려사항

Safety 기능의 핵심 목표 중 하나는 **장애물이 없는 정상 순찰에서 Safety가 멋대로 발동하지 않는 것**이다.

따라서 다음 항목들을 단계적으로 적용한다.

## 필수에 가까운 조건

- Navigation 상태에서만 Safety trigger 허용
- 전방 sector 제한
- 거리 threshold
- 최소 point 개수
- 여러 scan frame 연속 확인
- Static Map과 비교
- Static Map 비교 tolerance 적용

## 필요 시 추가할 수 있는 조건

- 현재 Nav2 Path와 장애물이 실제로 겹치는지 확인
- robot velocity가 실제 전진 중일 때만 trigger
- hysteresis 적용
- trigger 후 일정 시간 debounce / cooldown
- localization 신뢰도가 떨어진 상태에서는 Safety trigger 억제

처음부터 모든 조건을 넣지 말고, **실제 오작동이 발생하는 원인을 로그/RViz로 확인하면서 추가**한다.

---

# 21. Path까지 비교해야 하는가?

1차 버전에서는 필수 아님.

현재 목표는 전방에서 주행 진로를 막는 예상하지 못한 장애물을 안정적으로 검출하는 것이다.

우선:

```text
전방 + 가까움 + Static Map Free + 지속 검출
```

조건으로 충분히 시작할 수 있다.

만약 정상 주행 중 옆에 있는 신규 물체 때문에 불필요하게 Safety Stop이 발생한다면 다음 단계에서:

```text
신규 장애물 위치가 현재 Nav2 global/local path와 실제로 충돌하는가?
```

조건을 추가한다.

---

# 22. 개발 순서

## Phase 1 — 현재 코드 백업 및 Baseline 재확인

수정 전에 현재 상태를 보존한다.

확인:

- Bringup 정상
- `/scan` 정상
- AMCL 정상
- Nav2 정상
- A → B → C → D → E → A 한 바퀴 정상 주행

---

## Phase 2 — `followWaypoints()`를 waypoint별 `goToPose()`로 리팩터링

아직 Safety 기능을 넣지 말고 먼저:

```text
A → B → C → D → E → A
```

가 `goToPose()` 반복 구조에서도 기존과 동일하게 한 바퀴 정상 수행되는지 확인한다.

필수 변수:

```text
current_waypoint_index
current_waypoint_name
current_goal_pose
mission_state
```

---

## Phase 3 — Safety Monitor 단독 관찰 모드

아직 로봇을 멈추지 않는다.

Safety Monitor가 다음 정보만 로그로 출력하게 한다.

```text
front obstacle distance
front point count
static map comparison result
consecutive detection count
SAFETY_CANDIDATE 여부
```

정상 Waypoint 한 바퀴를 돌려 **false positive가 있는지 먼저 확인**한다.

이 단계가 중요하다.

정상 순찰에서 Safety Candidate가 반복 발생한다면 정지 동작을 붙이기 전에 먼저 조건을 수정한다.

---

## Phase 4 — Safety Stop + Nav2 Cancel만 테스트

아직 Scan/Escape를 하지 않는다.

```text
NAVIGATING
→ Safety Trigger
→ Stop
→ navigator.cancelTask()
→ 정지 상태 유지
```

검증:

- 정상 주행에서는 trigger가 발생하지 않는가
- 예상하지 못한 전방 장애물에서 trigger되는가
- Cancel 후 Nav2가 계속 주행 명령을 내리지 않는가
- 로봇이 확실하게 멈추는가

---

## Phase 5 — Scan Motion 추가

Cancel 이후에만 Scan Motion을 허용한다.

검증:

- 목표 각도에 안정적으로 도달하는가
- 회전 후 기준 heading으로 복귀하는가
- 실제 벽/장애물과 접촉하지 않는가
- Nav2와 `/cmd_vel` 경쟁이 없는가

---

## Phase 6 — Escape 추가

Scan Motion 이후 Escape 동작을 추가한다.

검증:

- 충분한 공간을 확보하면서 탈출하는가
- Escape 완료 판정이 안정적인가
- Escape 완료 시 속도가 0으로 정리되는가

---

## Phase 7 — 기존 Waypoint 재전송

Escape 완료 후:

```python
navigator.goToPose(current_goal_pose)
```

형태로 중단된 목적지를 다시 전송한다.

검증:

```text
B 이동 중 Safety 발생
→ Cancel
→ Safety/Scan/Escape
→ B 재전송
→ B 도착
→ C로 정상 진행
```

이 흐름이 성공해야 한다.

---

## Phase 8 — 반복 테스트 및 Parameter 튜닝

튜닝 대상 예:

```text
Safety 전방 각도
Safety 거리
Laser point 개수
연속 frame 수
Static map tolerance
회전 속도
Escape 속도
Escape 거리/종료 조건
Nav2 costmap inflation / robot radius 등
```

Costmap 파라미터는 Safety 기능 때문에 무조건 바꾸지 않고, 실제 문제가 확인된 항목만 조정한다.

---

## Phase 9 — 통합 Launch 작성

모든 기능이 개별적으로 안정화된 뒤 최종 시연용 launch를 만든다.

목표 예:

```bash
ros2 launch <project_package> factory_demo.launch.py
```

한 번의 launch로 필요한 요소를 실행하도록 구성:

```text
TurtleBot3 Bringup
LDS-03
Map Server
AMCL
Nav2
RViz2
Mission Manager
Safety Monitor / Recovery logic
```

단 개발 단계에서는 디버깅 편의를 위해 기존처럼 터미널을 분리해도 된다.

---

# 23. 개발 중 로그를 충분히 남길 것

Mission Manager에서는 상태 변화가 명확하게 보여야 한다.

예:

```text
[MISSION] NAVIGATING -> waypoint B
[SAFETY] candidate detected
[SAFETY] unexpected obstacle confirmed
[MISSION] NAVIGATING -> SAFETY_STOP
[NAV2] cancel current task
[NAV2] task canceled
[MISSION] SAFETY_STOP -> SCAN_LEFT
[MISSION] SCAN_LEFT -> SCAN_RIGHT
[MISSION] SCAN_RIGHT -> RETURN_HEADING
[MISSION] RETURN_HEADING -> ESCAPE
[MISSION] ESCAPE completed
[NAV2] resend waypoint B
[MISSION] RESEND_GOAL -> NAVIGATING
```

이 로그는 시연 실패 원인을 찾을 때 매우 중요하다.

---

# 24. 반드시 피할 구조

## 1. Nav2 전체 Stack 종료 후 다시 launch

하지 않는다.

```text
장애물
→ Nav2 Ctrl+C
→ Recovery
→ navigation2.launch.py 재실행
```

이 구조는 불필요하게 복잡하고 localization/navigation lifecycle 복구 문제가 커진다.

---

## 2. Nav2 Goal을 유지한 채 Recovery `/cmd_vel` 강제 우선순위 경쟁

초기 설계에서는 피한다.

```text
Nav2: 앞으로 가라
Recovery: 회전하라
Safety: 멈춰라
```

같은 명령 충돌이 생기지 않도록 한다.

---

## 3. `/scan`의 최소거리 1개만 보고 즉시 trigger

Noise와 정상 벽 때문에 false positive 가능성이 크다.

---

## 4. Static Map의 단일 cell만 보고 신규 장애물 판정

AMCL/scan/map discretization 오차 때문에 오검출 가능성이 있다.

---

## 5. Goal Cancel을 Waypoint 성공으로 처리

Safety로 Cancel되었으면 해당 Waypoint는 아직 완료된 것이 아니다.

반드시 같은 Waypoint를 다시 보내야 한다.

---

# 25. 발표/시연에서 사용할 용어

프로젝트 내부에서는 편의상 "E-stop"이라고 부를 수 있으나, 산업용 안전인증 하드웨어 Emergency Stop과 동일한 기능이라고 주장하지 않는다.

기술 설명에서는 다음 표현이 적절하다.

```text
LiDAR-based Software Safety Stop
Safety Stop + Escape Recovery
Unexpected Obstacle Detection and Recovery
```

전체 기능 설명 예:

> 저장된 전역지도 기반 Nav2 Waypoint 주행 중, 실시간 LiDAR 관측과 Static Map을 비교하여 예상하지 못한 전방 장애물을 검출한다. 위험 상황에서는 현재 Nav2 Navigation Task를 Cancel하고 Safety Stop 및 Scan/Escape 동작이 로봇 제어권을 넘겨받는다. 장애물 영역을 탈출한 후에는 중단 당시의 Waypoint를 Nav2에 다시 전송하여 현재 위치에서 경로를 재계산하고 순찰을 이어간다.

---

# 26. Codex가 다음 작업을 시작할 때 우선 확인할 파일

다음 파일을 먼저 읽고 현재 실제 코드가 이 문서와 일치하는지 확인한다.

```text
~/turtlebot3_ws/src/turtlebot3_waypoint_patrol/
  turtlebot3_waypoint_patrol/patrol_node.py

~/turtlebot3_ws/src/turtlebot3_waypoint_patrol/
  launch/patrol.launch.py

~/turtlebot3_ws/src/turtlebot3_navigation2/
  launch/navigation2.launch.py

~/turtlebot3_ws/src/turtlebot3_navigation2/
  param/humble/burger.yaml

~/turtlebot3_ws/maps/square_2m.yaml
~/turtlebot3_ws/maps/square_2m.pgm
```

필요하면 다음도 확인한다.

```text
~/turtlebot3_ws/src/turtlebot3_slam_toolbox/
~/turtlebot3_ws/src/coin_d4_driver/
~/turtlebot3_ws/src/turtlebot3_joystick/
```

---

# 27. Codex에게 전달할 작업 지시 요약

아래 방향을 기준으로 작업한다.

```text
1. 현재 patrol_node.py와 실제 실행 상태 확인

2. 기존 FollowWaypoints 한 번 호출 구조를
   waypoint별 goToPose() Mission State Machine으로 리팩터링

3. 리팩터링 상태에서 기존 A→B→C→D→E→A 한 바퀴가
   Safety 기능 없이 먼저 정상 동작하는지 검증

4. /scan + /map + TF를 이용한 Safety Monitor 추가

5. 정상 순찰에서 false positive가 없는지 관찰 모드로 검증

6. Safety Trigger 발생 시:
   - 즉시 Stop
   - 현재 Nav2 Task Cancel
   - Cancel 확인

7. Cancel 이후에만 Scan Motion / Escape가 로봇 제어

8. Escape 완료 후 current_goal_pose를 Nav2에 다시 전송

9. 해당 Waypoint 정상 도착 후에만 다음 waypoint로 이동

10. 충분한 반복 테스트 후 파라미터 튜닝

11. 마지막에 통합 launch 작성
```

---

# 28. 최종 목표 구조

```text
                        ┌─────────────────────────┐
                        │     Mission Manager     │
                        │                         │
                        │ current waypoint        │
                        │ mission state           │
                        │ safety state            │
                        └────────────┬────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
          Waypoint/Nav2        Safety Monitor        Scan / Escape
          Goal Manager         /scan + /map + TF     Recovery Control
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     │
                                     ▼
                                TurtleBot3
```

정상:

```text
Mission Manager → Nav2 → TurtleBot3
```

Safety 발생:

```text
Nav2 current task CANCEL
            ↓
Mission Manager → Safety/Scan/Escape → TurtleBot3
```

Escape 완료:

```text
Mission Manager
      ↓
중단된 Waypoint 재전송
      ↓
Nav2 현재 위치에서 Replan
      ↓
정상 Waypoint 순찰 재개
```

---

# 29. 최종 한 줄 원칙

> **Nav2를 죽이지 않는다. 현재 Navigation Goal/Task만 Cancel한다. Safety Stop/Scan/Escape가 제어권을 독점한다. Escape가 끝나면 기억해 둔 동일 Waypoint를 Nav2에 다시 보내고, 현재 위치에서 재계획하여 순찰을 계속한다.**
