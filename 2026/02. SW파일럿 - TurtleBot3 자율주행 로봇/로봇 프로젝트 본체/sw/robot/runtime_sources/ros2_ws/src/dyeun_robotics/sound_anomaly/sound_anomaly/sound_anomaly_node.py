"""Continuously classify microphone audio and drive the OpenCR status LEDs."""

from __future__ import annotations

import queue
import signal
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32, String, UInt8
import sounddevice as sd

from .audio_features import anomaly_probability, fixed_clip, mfcc_feature, resample_audio


class SoundAnomalyNode(Node):
    """Classify rolling audio windows as idle, normal, or abnormal."""

    def __init__(self) -> None:
        super().__init__("sound_anomaly_node")
        default_model = str(
            Path(get_package_share_directory("sound_anomaly"))
            / "models"
            / "gearbox_svm_source.joblib"
        )
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("audio_device", "")
        self.declare_parameter("capture_sample_rate", 0)
        self.declare_parameter("channels", 1)
        self.declare_parameter("prediction_interval", 1.0)
        self.declare_parameter("anomaly_threshold", -1.0)
        self.declare_parameter("silence_rms_threshold", 0.005)
        self.declare_parameter("led_topic", "/opencr_led_status")
        self.declare_parameter("red_blink_mode", 3)
        self.declare_parameter("green_blink_mode", 2)
        self.declare_parameter("idle_led_mode", 1)
        self.declare_parameter("shutdown_led_mode", 0)
        self.declare_parameter("led_heartbeat_period", 1.0)

        model_path = Path(self.get_parameter("model_path").value).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        saved_threshold = float(self.model.get("threshold", 0.50))
        configured_threshold = float(self.get_parameter("anomaly_threshold").value)
        self.threshold = saved_threshold if configured_threshold < 0.0 else configured_threshold
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("anomaly_threshold must be between 0 and 1, or -1 to use the model value")

        self.target_rate = int(self.model["sample_rate"])
        self.clip_seconds = float(self.model["clip_seconds"])
        self.n_mfcc = int(self.model["n_mfcc"])
        self.channels = int(self.get_parameter("channels").value)
        self.interval = float(self.get_parameter("prediction_interval").value)
        self.silence_threshold = float(self.get_parameter("silence_rms_threshold").value)
        self.red_blink_mode = int(self.get_parameter("red_blink_mode").value)
        self.green_blink_mode = int(self.get_parameter("green_blink_mode").value)
        self.idle_led_mode = int(self.get_parameter("idle_led_mode").value)
        self.shutdown_led_mode = int(self.get_parameter("shutdown_led_mode").value)
        for mode in (
            self.red_blink_mode,
            self.green_blink_mode,
            self.idle_led_mode,
            self.shutdown_led_mode,
        ):
            if not 0 <= mode <= 6:
                raise ValueError("OpenCR LED modes must be in the range 0..6")

        raw_device = str(self.get_parameter("audio_device").value)
        self.device = int(raw_device) if raw_device.isdigit() else (raw_device or None)
        configured_rate = int(self.get_parameter("capture_sample_rate").value)
        self.capture_rate = configured_rate or int(
            round(sd.query_devices(self.device, "input")["default_samplerate"])
        )
        self.window_samples = int(round(self.capture_rate * self.clip_seconds))
        self.hop_samples = int(round(self.capture_rate * self.interval))

        led_topic = str(self.get_parameter("led_topic").value)
        self.led_pub = self.create_publisher(UInt8, led_topic, 10)
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.probability_pub = self.create_publisher(Float32, "~/anomaly_probability", 10)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=40)
        self.buffer = np.empty(0, dtype=np.float32)
        self.samples_since_prediction = 0
        self.current_state = "IDLE"
        self.current_led_mode = self.idle_led_mode
        self.shutdown_led_sent = False
        self.shutdown_requested = False
        self.audio_stream_closed = False
        self._lock = threading.Lock()

        self._publish_state("IDLE", self.idle_led_mode, float("nan"))
        self.process_timer = self.create_timer(0.02, self._process_audio)
        heartbeat = float(self.get_parameter("led_heartbeat_period").value)
        self.heartbeat_timer = self.create_timer(heartbeat, self._publish_current_led)
        self.stream = sd.InputStream(
            samplerate=self.capture_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.get_logger().info(
            f"Started: capture={self.capture_rate}Hz, model={self.target_rate}Hz, "
            f"window={self.clip_seconds:.1f}s, threshold={self.threshold:.3f}"
        )

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.get_logger().warning(f"Audio input status: {status}")
        mono = np.mean(indata, axis=1, dtype=np.float32)
        try:
            self.audio_queue.put_nowait(mono.copy())
        except queue.Full:
            self.get_logger().warning("Audio queue full; dropping an input block")

    def _process_audio(self) -> None:
        if self.shutdown_requested:
            return
        received = False
        while True:
            try:
                incoming = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            received = True
            self.buffer = np.concatenate((self.buffer, incoming))
            self.samples_since_prediction += incoming.size
        if not received or self.buffer.size < self.window_samples:
            return
        self.buffer = self.buffer[-self.window_samples:]
        if self.samples_since_prediction < self.hop_samples:
            return
        self.samples_since_prediction = 0

        rms = float(np.sqrt(np.mean(np.square(self.buffer), dtype=np.float64)))
        if rms < self.silence_threshold:
            self._publish_state("IDLE", self.idle_led_mode, float("nan"))
            return
        try:
            audio = resample_audio(self.buffer, self.capture_rate, self.target_rate)
            samples = int(round(self.target_rate * self.clip_seconds))
            feature = mfcc_feature(fixed_clip(audio, samples), self.target_rate, self.n_mfcc)
            probability = anomaly_probability(self.model, feature)
            if probability >= self.threshold:
                self._publish_state("ABNORMAL", self.red_blink_mode, probability)
            else:
                self._publish_state("NORMAL", self.green_blink_mode, probability)
        except Exception as exc:  # Keep the background node alive and expose a safe state.
            self.get_logger().error(f"Inference failed: {exc}")
            self._publish_state("IDLE", self.idle_led_mode, float("nan"))

    def _publish_state(self, state: str, led_mode: int, probability: float) -> None:
        changed = state != self.current_state or led_mode != self.current_led_mode
        with self._lock:
            self.current_state = state
            self.current_led_mode = led_mode
        self.state_pub.publish(String(data=state))
        self.probability_pub.publish(Float32(data=probability))
        self.led_pub.publish(UInt8(data=led_mode))
        if changed:
            detail = "" if np.isnan(probability) else f", anomaly={probability:.3f}"
            self.get_logger().info(f"State={state}, LED mode={led_mode}{detail}")

    def _publish_current_led(self) -> None:
        with self._lock:
            mode = self.current_led_mode
        self.led_pub.publish(UInt8(data=mode))

    def destroy_node(self) -> bool:
        self._stop_audio_stream()
        self._publish_shutdown_led()
        return super().destroy_node()

    def _stop_audio_stream(self) -> None:
        if self.audio_stream_closed or not hasattr(self, "stream"):
            return
        self.audio_stream_closed = True
        self.stream.stop()
        self.stream.close()

    def _publish_shutdown_led(self) -> None:
        """Turn all LEDs off before this publisher and the OpenCR subscriber stop."""
        if self.shutdown_led_sent or not hasattr(self, "led_pub"):
            return
        self.shutdown_led_sent = True
        if rclpy.ok():
            self.led_pub.publish(UInt8(data=self.shutdown_led_mode))
            self.get_logger().info(f"Shutdown: LED mode={self.shutdown_led_mode}")
            # Allow DDS to hand the final command to turtlebot3_node.
            time.sleep(0.2)

    def request_shutdown(self) -> None:
        """Request an orderly shutdown from a Python signal handler."""
        self.shutdown_requested = True

def main(args=None) -> None:
    # Keep rclpy from closing the context before the final LED command is sent.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = SoundAnomalyNode()
        signal.signal(signal.SIGINT, lambda signum, frame: node.request_shutdown())
        signal.signal(signal.SIGTERM, lambda signum, frame: node.request_shutdown())
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.shutdown_requested:
            node._stop_audio_stream()
            node._publish_shutdown_led()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print(f"sound_anomaly_node failed: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
