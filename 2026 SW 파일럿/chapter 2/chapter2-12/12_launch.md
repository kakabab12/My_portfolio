# [과제 보고서] ROS2 Launch File 작성 및 다중 노드 자동화 제어

## 1. 수행 개요 및 목표
* **과제명**: ROS2 Chapter 2 - Problem 12 (Launch 파일 자동화)
* **환경**: Ubuntu 22.04 LTS, ROS2 Humble, Python 3.10
* **수행 목표**:
  1. ROS2의 Launch System 구조와 Python 기반 Launch 스크립트 작성법을 습득한다.
  2. `turtlesim_node`와 직전 과제에서 구현한 비동기 제어 노드(`turtle_move_control`)를 단일 명령어로 일괄 실행한다.
  3. 빌드 시스템(`colcon`)이 런치 파일을 인식하여 `install` 경로로 배포하도록 `setup.py`의 `data_files`를 올바르게 수정한다.
  4. 환경변수 충돌(Snap/GLIBC) 및 디스플레이 문제를 해결하는 디버깅 역량을 배양한다.

---

## 2. 관련 이론 및 주요 개념

### 2.1. ROS2 Launch System
ROS2의 Launch 시스템은 여러 개의 노드를 개별 터미널에서 일일이 구동할 필요 없이, 하나의 스크립트 실행만으로 시스템 전체 노드 그룹, 파라미터, 네임스페이스, 노드 간 의존성을 자동화하여 관리하는 도구이다. ROS2에서는 파이썬 API(`launch`, `launch_ros`)를 지원하여 조건부 실행, 이벤트 핸들링 등 유연한 제어가 가능하다.

### 2.2. `setup.py` 배포 설정의 필요성
ROS2 파이썬 패키지 구조에서 소스 디렉터리(`src/my_robot_controller/launch/`)에 작성된 런치 파일은 빌드 시 자동으로 실행 경로로 복사되지 않는다.  
따라서 `setup.py` 내 `data_files` 리스트에 설치 경로(`share/<package_name>/launch`)를 정의해 주어야 `colcon build` 실행 시 `install/share/my_robot_controller/launch/`로 파일이 배포되어 `ros2 launch` 명령어로 접근할 수 있게 된다.

---

## 3. 구현 내용 및 코드

### 3.1. `turtle_move.launch.py` (런치 스크립트)
> **파일 경로**: `~/ros2_ws/src/my_robot_controller/launch/turtle_move.launch.py`

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Turtlesim 시뮬레이터 노드
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim'
        ),
        # 2. 자율 주행 및 비동기 서비스 제어 노드
        Node(
            package='my_robot_controller',
            executable='turtle_move_control',
            name='turtle_move_control'
        )
    ])