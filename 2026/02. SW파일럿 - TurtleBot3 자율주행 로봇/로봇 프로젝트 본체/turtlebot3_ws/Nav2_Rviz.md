## 터미널 1 — TurtleBot3 본체

  source /opt/ros/humble/setup.bash
  source ~/turtlebot3_ws/install/setup.bash
  export TURTLEBOT3_MODEL=burger
  export LDS_MODEL=LDS-03

  ros2 launch turtlebot3_bringup robot.launch.py

  계속 실행해 둡니다.

## 터미널 2 — Nav2와 RViz

  새 터미널을 열고 실행합니다.

  source /opt/ros/humble/setup.bash
  source ~/turtlebot3_ws/install/setup.bash
  export TURTLEBOT3_MODEL=burger

  ros2 launch turtlebot3_navigation2 navigation2.launch.py \
    map:=$HOME/turtlebot3_ws/maps/square_1_5m.yaml

  그러면 Nav2와 설정된 RViz 창이 자동으로 열립니다.

  RViz가 열리면 바로 목표를 지정하지 말고 먼저:

  1. 상단의 2D Pose Estimate 선택
  2. 지도에서 실제 로봇 위치를 클릭
  3. 마우스를 로봇이 바라보는 방향으로 드래그
  4. 라이다의 빨간 점과 지도 벽이 겹치는지 확인
  5. 필요하면 초기 위치 설정을 반복

  위치가 맞으면 Navigation2 Goal로 가까운 위치부터 시험합니다.

