import rclpy
from rclpy.node import Node

class LoggingNode(Node):
    def __init__(self):
        super().__init__('logging_node')
        self.get_logger().info('Logging node has been started!')

def main(args=None):
    rclpy.init(args=args)
    node = LoggingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()