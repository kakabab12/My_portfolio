# 8_publisher.md

## 1. 노드 간 통신 구조 정리
- **turtlesim_node**: 토픽을 구독(Subscribe)하여 제어 명령을 받고 수행하는 역할.
- **turtle_teleop_key**: 키보드 입력을 받아 토픽으로 게시(Publish)하는 퍼블리셔 역할.
- **circle_turtle_node (구현)**: 직접 구현한 노드로, `geometry_msgs/msg/Twist` 메시지를 `/turtle1/cmd_vel` 토픽으로 0.1초마다 게시하여 원을 그리며 이동하게 함.

## 2. 퍼블리셔 구현 상세
- `geometry_msgs/msg/Twist` 메시지의 `linear.x` (전진)와 `angular.z` (회전) 값을 적절히 조절하여 원 궤적을 생성했습니다.
- `package.xml`에 `geometry_msgs`와 `turtlesim` 의존성을 추가하여 메시지 타입과 터틀심 패키지를 정상적으로 참조하도록 설정했습니다.

## 3. 실습 결과
- `rqt_graph`를 통해 `circle_turtle_node`가 게시자로서 `/turtle1/cmd_vel` 토픽을 발행하고 있음을 확인했습니다.


![rqt_graph 결과](./Screenshot%20from%202026-07-11%2014-17-54.png)