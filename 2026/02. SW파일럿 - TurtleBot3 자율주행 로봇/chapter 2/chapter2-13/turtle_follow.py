import math
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, Kill
from std_srvs.srv import Empty


class TurtleFollowNode(Node):
    def __init__(self):
        super().__init__('turtle_follow_control')

        # 상태 변수
        self.turtle1_pose = None
        self.turtle2_pose = None
        self.spawn_complete = False
        self.is_shutting_down = False

        # 서비스 클라이언트 생성 (/spawn, /kill)
        self.spawn_client = self.create_client(Spawn, 'spawn')
        self.kill_client = self.create_client(Kill, 'kill')

        # 서비스 서버 생성 (/quit)
        self.quit_service = self.create_service(Empty, 'quit', self.quit_callback)

        self.get_logger().info('Turtle Follow Node가 시작되었습니다.')
        
        # turtle2 생성 요청 실행
        self.spawn_turtle2()

    def spawn_turtle2(self):
        """/spawn 서비스를 비동기로 호출하여 turtle2를 (2.0, 2.0) 위치에 생성"""
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/spawn 서비스를 기다리는 중...')

        req = Spawn.Request()
        req.x = 2.0
        req.y = 2.0
        req.theta = 0.0
        req.name = 'turtle2'

        future = self.spawn_client.call_async(req)
        future.add_done_callback(self.spawn_response_callback)

    def spawn_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"성공적으로 '{response.name}'를 (2.0, 2.0)에 생성했습니다.")
            self.setup_publishers_and_subscribers()
        except Exception as e:
            self.get_logger().error(f'turtle2 생성 실패: {e}')

    def setup_publishers_and_subscribers(self):
        """위치 구독자 및 속도 발행자, 제어 타이머 설정"""
        self.sub_t1 = self.create_subscription(Pose, '/turtle1/pose', self.t1_pose_cb, 10)
        self.sub_t2 = self.create_subscription(Pose, '/turtle2/pose', self.t2_pose_cb, 10)

        # turtle2 이동 제어 토픽 발행자
        self.cmd_pub_t2 = self.create_publisher(Twist, '/turtle2/cmd_vel', 10)

        # 20Hz (0.05초) 제어 루프 타이머
        self.timer = self.create_timer(0.05, self.control_loop)
        self.spawn_complete = True

    def t1_pose_cb(self, msg):
        self.turtle1_pose = msg

    def t2_pose_cb(self, msg):
        self.turtle2_pose = msg

    def control_loop(self):
        """turtle2가 turtle1을 자연스럽게 추적하는 P 제어 루프"""
        if not self.spawn_complete or self.is_shutting_down:
            return
        if self.turtle1_pose is None or self.turtle2_pose is None:
            return

        # 두 로봇 간의 거리 및 상대 각도 계산
        dx = self.turtle1_pose.x - self.turtle2_pose.x
        dy = self.turtle1_pose.y - self.turtle2_pose.y
        distance = math.hypot(dx, dy)

        # 1. 두 로봇이 만났을 때 (거리 0.8 이하)
        if distance < 0.8:
            self.get_logger().info('두 거북이가 만났습니다! turtle2를 정지 후 삭제하고 노드를 종료합니다.')
            self.is_shutting_down = True
            self.stop_turtle2()
            self.kill_turtle('turtle2', shutdown_after=True)
            return

        # 2. P 제어 기반 궤적 추적
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - self.turtle2_pose.theta
        # 각도 오차를 [-pi, pi] 범위로 정규화
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        cmd = Twist()
        cmd.linear.x = min(1.5, 1.2 * distance)   # 거리에 비례한 선속도 (최대 1.5m/s)
        cmd.angular.z = 4.0 * angle_error         # 각도 오차에 비례한 각속도
        self.cmd_pub_t2.publish(cmd)

    def stop_turtle2(self):
        cmd = Twist()
        self.cmd_pub_t2.publish(cmd)

    def kill_turtle(self, name, shutdown_after=False):
        """/kill 서비스를 호출하여 거북이 제거"""
        if not self.kill_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/kill 서비스를 찾을 수 없습니다.')
            if shutdown_after:
                self.destroy_node()
                rclpy.shutdown()
            return

        req = Kill.Request()
        req.name = name
        future = self.kill_client.call_async(req)

        def kill_cb(fut):
            try:
                fut.result()
                self.get_logger().info(f"성공적으로 '{name}'를 제거했습니다.")
            except Exception as e:
                self.get_logger().error(f"'{name}' 제거 실패: {e}")

            if shutdown_after:
                self.get_logger().info('시스템을 안전하게 종료(Clean Shutdown)합니다.')
                self.destroy_node()
                rclpy.shutdown()

        future.add_done_callback(kill_cb)

    def quit_callback(self, request, response):
        """/quit 서비스 호출 시 두 거북이 모두 제거 후 클린 셧다운"""
        self.get_logger().info('/quit 서비스 호출 수신: 모든 거북이를 제거하고 노드를 종료합니다.')
        self.is_shutting_down = True
        if hasattr(self, 'timer'):
            self.timer.cancel()

        # turtle1 및 turtle2 순차 제거 후 종료
        self.kill_turtle('turtle1', shutdown_after=False)
        self.kill_turtle('turtle2', shutdown_after=True)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TurtleFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()