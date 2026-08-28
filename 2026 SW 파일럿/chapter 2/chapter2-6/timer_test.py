import rclpy
from rclpy.node import Node

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')
        self.counter = 0
        
        # 2초 타이머 생성
        self.timer2 = self.create_timer(2.0, self.timer_callback_2s)
        # 3초 타이머 생성
        self.timer3 = self.create_timer(3.0, self.timer_callback_3s)

    def timer_callback_2s(self):
        self.counter += 1
        self.get_logger().info(f'2 seconds passed : {self.counter}')

    def timer_callback_3s(self):
        self.counter -= 1
        self.get_logger().info(f'3 seconds passed : {self.counter}')

def main(args=None):
    rclpy.init(args=args)
    node = TimerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()