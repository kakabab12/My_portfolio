import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .controller import LedController, VALID_STATES


class GpioBackend:
    def __init__(self, green_pin, red_pin, active_high=True):
        try:
            import Jetson.GPIO as GPIO
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                'Jetson.GPIO를 사용할 수 없습니다. /dev/gpiochip 권한과 설치 상태를 확인하세요.'
            ) from exc
        self.gpio = GPIO
        self.green_pin = green_pin
        self.red_pin = red_pin
        self.on = GPIO.HIGH if active_high else GPIO.LOW
        self.off = GPIO.LOW if active_high else GPIO.HIGH
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(green_pin, GPIO.OUT, initial=self.off)
        GPIO.setup(red_pin, GPIO.OUT, initial=self.off)

    def write(self, green, red):
        self.gpio.output(self.green_pin, self.on if green else self.off)
        self.gpio.output(self.red_pin, self.on if red else self.off)

    def close(self):
        self.write(False, False)
        self.gpio.cleanup((self.green_pin, self.red_pin))


class MockBackend:
    def __init__(self, logger):
        self.logger = logger
        self.last = None

    def write(self, green, red):
        value = (green, red)
        if value != self.last:
            self.logger.info(f'[MOCK] green={int(green)} red={int(red)}')
            self.last = value

    def close(self):
        self.last = (False, False)


class LedStatusNode(Node):
    def __init__(self):
        super().__init__('led_status_node')
        self.declare_parameter('green_pin', 31)
        self.declare_parameter('red_pin', 33)
        self.declare_parameter('blink_period', 0.5)
        self.declare_parameter('active_high', True)
        self.declare_parameter('mock', False)
        self.declare_parameter('initial_state', 'idle')

        mock = self.get_parameter('mock').value
        if mock:
            self.backend = MockBackend(self.get_logger())
        else:
            self.backend = GpioBackend(
                self.get_parameter('green_pin').value,
                self.get_parameter('red_pin').value,
                self.get_parameter('active_high').value,
            )

        self.controller = LedController(self.backend.write)
        initial = self.get_parameter('initial_state').value
        self.controller.set_state(initial)
        period = float(self.get_parameter('blink_period').value)
        if period <= 0:
            raise ValueError('blink_period must be greater than 0')
        self.timer = self.create_timer(period, self.controller.tick)
        self.subscription = self.create_subscription(
            String, 'led_status', self.status_callback, 10
        )
        self.get_logger().info(
            f'ready: mode={"mock" if mock else "GPIO.BOARD"}, '
            f'states={", ".join(VALID_STATES)}'
        )

    def status_callback(self, msg):
        state = msg.data.strip().lower()
        try:
            self.controller.set_state(state)
            self.get_logger().info(f'state -> {state}')
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def destroy_node(self):
        self.backend.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LedStatusNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as exc:
        if node is not None and rclpy.ok():
            node.get_logger().error(str(exc))
        else:
            print(f'led_status_node: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
